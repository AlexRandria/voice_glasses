"""base"""
import numpy as np 
import pandas as pd 

"""tensorflow pour modèle et formatage entrée"""
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras
from tensorflow.keras import layers

""" Matplotlib pour graphiques """
import matplotlib.pyplot as plt

import os 
    
    
""" Fonction responsable de l'architechture du modèle """
def Recup_model_non_compile(output_dim, input_dim=257, nb_couches_rnn=5, taille_rnn=512):
    """Model similar to DeepSpeech2."""
    # Input du modèle
    input_spectrogram = layers.Input((None, input_dim), name="input")
    
    # Augmentation des dimensions pour pouvoir utiliser les CNN
    x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
    
    # 1ere couche de Convolution
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 41],
        strides=[2, 2],
        padding="same",
        use_bias=False,
        name="conv_1",
    )(x)
    x = layers.BatchNormalization(name="conv_1_bn")(x)
    x = layers.ReLU(name="conv_1_relu")(x)
    
    # 2nde couche de Convolution 
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 21],
        strides=[1, 2],
        padding="same",
        use_bias=False,
        name="conv_2",
    )(x)
    x = layers.BatchNormalization(name="conv_2_bn")(x)
    x = layers.ReLU(name="conv_2_relu")(x)
    
    # Reshape le résultat pour pouvoir alimenter les couches RNN
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    
    # Couches RNN 
    for i in range(1, nb_couches_rnn + 1):
        recurrent = layers.GRU(
            units=taille_rnn,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
        )
        x = layers.Bidirectional(
            recurrent, name=f"bidirectional_{i}", merge_mode="concat"
        )(x)
        if i < nb_couches_rnn:
            x = layers.Dropout(rate=0.5)(x)
            
    # Couche Dense 
    x = layers.Dense(units=taille_rnn * 2, name="dense_1")(x)
    x = layers.ReLU(name="dense_1_relu")(x)
    x = layers.Dropout(rate=0.5)(x)
    
    # Couches de classification
    output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
    
    # Model
    model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")
    return model



""" Définition de la classe de la fonction de perte (CTC) """
class CTCLoss(keras.losses.Loss):
    def __init__(self, name="CTCLoss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # Taille de batch
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        
        # Longeur de y_pred
        y_pred_len = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        
        # Longeur de y_true 
        y_true_len = tf.cast(tf.shape(y_true)[1], dtype="int64")

        # Reformatage des données
        y_pred_len = y_pred_len * tf.ones(shape=(batch_len, 1), dtype="int64")
        y_true_len = y_true_len * tf.ones(shape=(batch_len, 1), dtype="int64")
        
        # Calcul de la perte
        loss = keras.backend.ctc_batch_cost(y_true, y_pred, y_pred_len, y_true_len)
        
        return loss
    
    
""" Définition de la classe metric CER (Taux d'erreur sur caractère) """
class CER_metric(tf.keras.metrics.Metric):
    def __init__(self, name='CER_metric', **kwargs):
        super(CER_metric, self).__init__(name=name, **kwargs)
        
        # Définition des variables de la class
        self.cumul_CER = tf.Variable(0.0, name="cumul_CER", dtype=tf.float32)
        self.compteur_longeur = tf.Variable(0, name="compteur_batch", dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Shape et longeur de y_pred
        y_pred_shape = tf.keras.backend.shape(y_pred)
        y_pred_len = tf.ones(shape=y_pred_shape[0]) * tf.keras.backend.cast(y_pred_shape[1], "float32")

        # Décodage de la prédiction via décodeur ctc
        decode, _ = tf.keras.backend.ctc_decode(y_pred,
                                    y_pred_len,
                                    greedy=True)

        # Passage en sparse tensor
        decode = tf.keras.backend.ctc_label_dense_to_sparse(decode[0], tf.keras.backend.cast(y_pred_len,"int32"))
        y_true_sparse = tf.keras.backend.ctc_label_dense_to_sparse(y_true, tf.keras.backend.cast(y_pred_len,"int32"))

        # On supprime les valeurs éguale à -1 correspond à la valeur dù au padding de ctc_decode
        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        
        # Calcul de la distance de Levenshtein entre la prédiction et le réel
        distance = tf.edit_distance(decode, y_true_sparse, normalize=True)

        # Stockage de la distance Levenshtein cumulée et du compteur de longeur de la transcription de référence 
        self.cumul_CER.assign_add(tf.reduce_sum(distance))
        self.compteur_longeur.assign_add(len(y_true))

    def result(self):
        # Retourne le CER (distance cumulé pour les caractères divisée par taille chaîne)
        return tf.math.divide_no_nan(self.cumul_CER, tf.keras.backend.cast(self.compteur_longeur, "float32"))

    def reset_state(self):
        # Reset des variables à chaque epoch
        self.cumul_CER.assign(0.0)
        self.compteur_longeur.assign(0)
        

        
""" Fonction de compilation du modèle """        
def Recup_model_compile(output_dim):
    model = Recup_model_non_compile(output_dim=output_dim)
    
    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    
    # Compilation du modèle
    model.compile(optimizer=opt, 
                  loss=CTCLoss(), 
                  metrics = [CER_metric()]
                 )
    return model


""" Fonctiond de fabriquation ou restoration du modèle """
def Fabriquer_ou_restorer_model(output_dim, checkpoint_doss, name):
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_doss + name]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(filepath = latest_checkpoint, 
                                       custom_objects = {"CTCLoss":CTCLoss,
                                                         "CER_metric":CER_metric},
                                       compile=False)
    print("Creating a new model")
    return Recup_model_compile(output_dim=output_dim)


""" Sauvegarde de l'historique de l'entraînement dans un fichier csv """
def sauvegarde_history_file(History):
    # Définition du nom du fichier pour sauvegarder l'historique
    history_filename = "history.csv"

    # Sauvegarde de l'historique dans un fichier CSV sans effacer les versions précédentes
    if os.path.exists(history_filename):
        previous_history = pd.read_csv(history_filename)
        history = pd.DataFrame(History.history)
        updated_history = pd.concat([previous_history, history], ignore_index=True)
        updated_history.to_csv(history_filename, index=False)
    else:
        history = pd.DataFrame(History.history)
        history.to_csv(history_filename, index=False)
        
        
        
""" Visualisation des métriques d'entraînement """
def analyze_metrics(history):
    fig = plt.figure(figsize = (8,4))     
    
    ax1 = fig.add_subplot(121)
    ax1.plot(history.history['loss'], label='Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Valeur loss par epoch')
    ax1.legend()
    
    ax2 = fig.add_subplot(122)
    ax2.plot(history.history['CER_metric'], label='CER_metric')
    ax2.plot(history.history['val_CER_metric'], label='Validation CER_metric')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Valeur loss par epoch')
    ax2.legend()
    
    plt.title('Métriques d\'entraînement')
    plt.show()