import boto3
import sys
import logging
import pymysql
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio
from Module_reco_voc_2 import Fabriquer_ou_restorer_model
import keras
from jiwer import wer, cer
import os

# rds settings
user_name = os.environ['USER_NAME']
password = os.environ['PASSWORD']
rds_host = os.environ['RDS_HOST']
db_name = os.environ['DB_NAME']

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# create the database connection outside of the handler to allow connections to be
# re-used by subsequent function invocations.
try:
    conn = pymysql.connect(host=rds_host, user=user_name, passwd=password, db=db_name, connect_timeout=5)
except pymysql.MySQLError as e:
    logger.error("ERROR: Unexpected error: Could not connect to MySQL instance.")
    logger.error(e)
    sys.exit()

logger.info("SUCCESS: Connection to RDS MySQL instance succeeded")

# create connexion s3
s3 = boto3.client('s3')

# Liste des caractères acceptés
caracteres = [x for x in "abcdefghijklmnopqrstuvwxyz' "]

# Mapping des caractères en chiffres (int)
char_to_num = keras.layers.StringLookup(vocabulary=caracteres, oov_token="")

# Mapping (retour) des chiffres à des caractères
num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# Préparation de l'enregistrement
frame_length = 512
frame_step = 128
fft_length = 512

def lambda_handler(event, context):
    # TODO implement
    id_dataset = event["id_dataset"]
    id_record = event["id_record"]
    phrase = event["transcription_real"]

    if len(id_dataset)>0:

        # Récupération de l'enregistrement en bdd
        with conn.cursor() as cur:
            sql = "SELECT * FROM dataset WHERE id_audio = '%s'" % (id_dataset)
            print(sql)
            cur.execute(sql)
            output = cur.fetchall()
        conn.commit()
        df_predict = pd.DataFrame(output, columns=['id_audio', 'speaker_id', 'chapter_id', 'id_line',
                                                   'chemin', 'transcription', 'flag_train'])
        print(df_predict)

        # datasets/train-clean-100/1069/133709/1069-133709-0000.flac
        # sources/librispeech/train-clean-360/records/
        url_audio = df_predict.iloc[0, :]['chemin']
        url_audio = url_audio.replace("sources/librispeech/", "datasets/").replace("records/", "")
        print("> ouverture fichier %s" % url_audio)
        fic_audio = s3.get_object(Bucket='voiceglasses', Key=url_audio)

    else:

        fic_audio = s3.get_object(Bucket='voiceglasses', Key="datasets/record/%s.flac" % id_record)


    print("> Copie du fichier audio en local (/tmp/audio.flac)")
    audio = fic_audio['Body']
    audio = audio.read()

    print("> Copie du fichier audio en local (/tmp/audio.flac)")
    f = open('/tmp/audio.flac', 'wb')
    f.write(audio)
    f.close()
    print("> Fichier audio copié en local (/tmp/audio.flac)")

    # Récupération du modèle #models/v1/recovocale_final.hdf5
    if not os.path.exists('/tmp/model_v1.hdf5'):
        fic_model = s3.get_object(Bucket='voiceglasses', Key="models/v1/recovocale_final.hdf5")
        print(">Récupération model sur s3")
        h = open('/tmp/model_v1.hdf5', 'wb')
        h.write(fic_model['Body'].read())
        h.close()

    model = Fabriquer_ou_restorer_model(output_dim=char_to_num.vocabulary_size(),
                                        checkpoint_doss='/tmp/',
                                        name='model_v1.hdf5')

    batch_size = 28
    print("> Init du dataset")
    if len(id_dataset) > 0:
        predict_dataset = tf.data.Dataset.from_tensor_slices(
            (list(['/tmp/audio.flac']), list(df_predict["transcription"])))
    else:
        predict_dataset = tf.data.Dataset.from_tensor_slices(
            (list(['/tmp/audio.flac']), list([phrase])))
    print(predict_dataset)

    predict_dataset = (predict_dataset.map(Recup_spectrogramme_transcription, num_parallel_calls=tf.data.AUTOTUNE)
                       .padded_batch(batch_size)
                       # .prefetch(buffer_size=tf.data.AUTOTUNE)
                       )

    # Prédictions
    predictions = []
    targets = []

    for batch in predict_dataset:
        X, y = batch
        batch_predictions = model.predict(X)
        batch_predictions = decode_batch_predictions(batch_predictions)
        predictions.extend(batch_predictions)
        for label in y:
            label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
            targets.append(label)

    print("#### Phrase prononcée")
    print(targets[0])

    print("### Phrase prédite")
    print(predictions[0])
    print("Phrase prédite complète")
    print(predictions)

    print("### Evaluation de l'erreur")

    print("Character Error Rate")
    cer_metric = cer(targets[0], predictions[0])
    print("### CER = %s %%" % cer_metric)

    print("Word Error Rate")
    wer_metric = wer(targets[0], predictions[0])
    print("### WER = %s %%" % wer_metric)

    # Récupération du seuil de confiance et ajout de la prédiction dans la table predict
    with conn.cursor() as cur:
        cur.execute("select * from param")
        output = cur.fetchone()
        seuil_confiance = output[0] / 100

        cur.execute("create table if not exists predict ( id_predict INT NOT NULL AUTO_INCREMENT, id_record int NOT NULL, Timestamp datetime default CURRENT_TIMESTAMP, transcription_reel varchar(255) NOT NULL, transcription_predict varchar(255) NOT NULL, cer float NOT NULL, wer float NOT NULL, PRIMARY KEY (id_predict))")
        sql = f"INSERT INTO predict (id_record, transcription_reel, transcription_predict, cer, wer) values ('{id_record}', '{phrase}', '{predictions[0]}', '{cer_metric}', '{wer_metric}')"
        cur.execute(sql)
    conn.commit()

    # test qualité
    bTrain = False
    if len(id_record)>0 & (cer_metric < seuil_confiance) :
        with conn.cursor() as cur:
            cur.execute("create table if not exists dataset_record ( id_audio varchar(50) NOT NULL, transcription varchar(500) NOT NULL, flag_train boolean, PRIMARY KEY (id_audio))")
            cur.execute(f"insert into dataset_record (id_audio, transcription, flag_train) values(%s,%s,%s)", (id_record, phrase, False))            
        conn.commit()
        bTrain = True

    return {
        'statusCode': 200,
        'transcription_real': targets[0],
        'transcription_predict': predictions[0],
        'cer': cer_metric,
        'wer': wer_metric,
        'train': bTrain
    }

def Recup_spectrogramme_transcription(fichier_audio, transcription):
    fichier = tf.io.read_file(fichier_audio)
    audio = tfio.audio.decode_flac(fichier, dtype=tf.int16)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)  # pas utile dans notre cas
    spectrogram = tf.signal.stft(audio,
                                    frame_length=frame_length,
                                    frame_step=frame_step,
                                    fft_length=fft_length
                                    )
    spectrogram = tf.abs(spectrogram)  # valeur absolue
    spectrogram = tf.math.pow(spectrogram, 0.5)  # racine carrée
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    transcription = tf.strings.lower(transcription)
    transcription = tf.strings.unicode_split(transcription, input_encoding="UTF-8")
    transcription = char_to_num(transcription)
    return spectrogram, transcription

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text
