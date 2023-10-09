"""base"""
import numpy as np 
import pandas as pd 

"""gestion du temps et des fichiers"""
from time import time    

from datetime import datetime

import os       

from os import walk

""" Affichage et image """
import IPython.display as ipd  

"""tensorflow pour modèle et formatage entrée"""
import tensorflow_io as tfio

""" construction du jeu de donnée """
from sklearn.model_selection import train_test_split


""" Module des fonctions perso pour le modèle de reconnaissance vocale """

""" Recuperation des meta données et des transcriptions (data text) """

""" Partie meta data """
def Lecture_fichier_CHAPTERS(fichier="open_SLR_data/CHAPTERS_data.csv"):
    # Ouverture du fichier chapter
    CHAPTERS = pd.read_csv(fichier,delimiter='|')
    
    # Suppression des colonnes inutiles
    CHAPTERS = CHAPTERS[['subset','speaker_id','chapter_id']]
    return CHAPTERS
    
    
def Lecture_fichier_SPEAKERS(fichier="open_SLR_data/SPEAKERS_data.csv"):
    # Ouverture du fichier chapter
    SPEAKERS = pd.read_csv(fichier,delimiter='|')
    
    # Suppression des colonnes inutiles
    SPEAKERS = SPEAKERS[['speaker_id','gender']]
    return SPEAKERS
    
    
def Fusion_fichiers_CHAPTERS_SPEAKERS(fic_chapters="open_SLR_data/CHAPTERS_data.csv", 
                                      fic_speakers="open_SLR_data/SPEAKERS_data.csv"):
    
    CHAPTERS = Lecture_fichier_CHAPTERS(fic_chapters)
    SPEAKERS = Lecture_fichier_SPEAKERS(fic_speakers)
    
    meta = pd.merge(SPEAKERS, CHAPTERS, on="speaker_id")

    meta = meta[['subset','speaker_id', 'chapter_id', 'gender']]

    return meta
    
    

""" Partie transcriptions """


def lecture_fic_txt(fichier, Repertoire):
    # ouverture du fichier .txt de la forme non
    text_file = open(str(Repertoire
                         +"/"
                         +fichier.split("-")[0]
                         +"/"
                         +fichier.split("-")[1].split(".")[0]
                         +"/"
                         +fichier),'r')
    # stockage du contenue dans une liste
    lines = text_file.readlines()
    # fermeture du fichier
    text_file.close()
    return lines
    
    
    
def enregistrement_txt(lines,entetes,monRepertoire,type_fic_audio):
    val_txt = []
    
    for line in lines:
        id_audio = line[:line.find(" ")]
        id_speaker = line[:line.find("-")]
    
        id_2 = line[line.find("-")+1:len(id_audio)]
        id_chapter = id_2[:id_2.find("-")]
        id_line = str(id_2[id_2.find("-")+1:len(id_2)])
    
        transcription = line[line.find(" ")+1:-1]
        
        chemin = str(monRepertoire+"/"+id_speaker+"/"+id_chapter+"/"+id_audio+type_fic_audio)
    
        val_txt.append([id_audio,id_speaker,id_chapter,id_line,chemin,transcription])
        
    df_txt = pd.DataFrame(val_txt,columns=entetes)
    return df_txt
    
    
def recuperation_transcription(monRepertoire,type_fic_audio, bSaveCsv=True):
    listeFichiers_txt = []
    listeFichiers_error = []
    nb_error = 0
    nb_fic_audio = 0
    entetes = ['id_audio','speaker_id','chapter_id','id_line','chemin','transcription']
    transcriptions = pd.DataFrame(columns=entetes)

    # Lecture du dossier monRepertoire et de l'ensemble de ses sous-dossiers et des fichiers
    for (repertoire, sousRepertoires, fichiers) in walk(monRepertoire):
        for fichier in fichiers :
            if ".txt" in fichier :
                listeFichiers_txt.append(fichier)
                lines = lecture_fic_txt(fichier,monRepertoire)
                df_txt = enregistrement_txt(lines,entetes,monRepertoire,type_fic_audio)
            
                transcriptions = pd.concat([transcriptions,df_txt])
        
            elif type_fic_audio in fichier :
                nb_fic_audio +=1
            
            else :
                listeFichiers_error.append(fichier)

    if nb_error != 0 :
        print("Nombres d'erreures :",nb_error)
        display(listeFichiers_error)
    
    if transcriptions.shape[0] == nb_fic_audio:
        print("Nombre de fichier audio enregistrés :",nb_fic_audio)
        if bSaveCsv:
            transcriptions.to_csv('open_SLR_data/text_data.csv', index=False)
    else:
        print("Erreur : Nombre de fichier audio lus({}) différent du nombre de lignes enregistrées ({})".format(nb_fic_audio,transcriptions.shape[0]))

    return transcriptions



""" Partie Union """

def Union_meta_transcription(meta, transcriptions):
    meta_end = meta[['speaker_id','gender']]

    meta_end = meta_end.astype({'speaker_id':str})

    meta_end = meta_end.drop_duplicates()
    
    transcriptions = transcriptions.merge(right=meta_end, on="speaker_id", how='inner')

    transcriptions = transcriptions[['id_audio','speaker_id','gender','chapter_id','id_line','chemin','transcription']]

    transcriptions.to_csv('open_SLR_data/text_data.csv', index=False)
    return transcriptions
    
    
""" Partie globale """

def Recuperation_data_txt(fic_chapters="open_SLR_data/CHAPTERS_data.csv", 
                          fic_speakers="open_SLR_data/SPEAKERS_data.csv",
                          monRepertoire = 'train-clean-100',
                          type_fic_audio = ".flac"):
    
    meta = Fusion_fichiers_CHAPTERS_SPEAKERS(fic_chapters,fic_speakers)
    
    transcriptions = recuperation_transcription(monRepertoire,type_fic_audio)
    
    transcriptions = Union_meta_transcription(meta,transcriptions)
    
    return transcriptions



""" Partie Audio-time (librosa) """

def Recuperation_audio_times(audio_files, sr=16000, sauvegarde = False):
    audio_times = []
    for i in range(audio_files.shape[0]):
        # Récupération de Y
        y, sr = librosa.load(audio_files[i])
        
        if sauvegarde:
            # Sauvegarde de Y
            sauvegarder_audio_time(y,audio_files[i])
        
        # Ajout de Y à la liste pour suite traitement
        audio_times.append(y)
    return audio_times

    
def sauvegarder_audio_time(y,file):
    file=file.split(".")[0]+".npy"
    
    with open(file, 'wb') as f:
        np.save(f, y)    
    
    
def charger_audio_time(file):
    file=file.split(".")[0]+".npy"
    
    with open(file, 'rb') as f:
        y = np.load(f)
    return y    
    
    
""" Partie durée audio (librosa) """

def Enregistrement_duree(duree, fichier='open_SLR_data/text_data.csv'):
    # récupération du dataframe
    transcriptions = pd.read_csv('../pages/open_SLR_data/text_data.csv')
    
    # Controle de l'existance de la colonne duree
    if 'duree' in(transcriptions):
        # Identification de la première ligne vide
        a = len(transcriptions['duree'])-(pd.isna(transcriptions['duree'])).sum()
        # valorisation de n ligne de la colonne duree
        transcriptions.loc[a:a+len(duree)-1, 'duree']=duree
        
    else :
        # création de la nouvelle colonne vide
        transcriptions['duree']=np.nan
    
        # valorisation de n ligne de la colonne duree
        transcriptions.loc[:len(duree)-1,'duree']=duree
    
    # sauvegarde du dataframe à jour
    transcriptions.to_csv('open_SLR_data/text_data.csv', index=False)
    
    return transcriptions


""" Fonction d'automatisation du traitement de l'audio """

def preprocess(fichier='open_SLR_data/text_data.csv', nb_batch=-1, sauvegarde = False):
    # Heure de lancement de la fonction
    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    print("Heure de lancement :",date_time)

    # récupération du dataframe
    transcriptions = pd.read_csv('../pages/open_SLR_data/text_data.csv')
    
    # Calcul du nombre de lignes à traiter
    if nb_batch ==-1:
        # Nombre de ligne à traiter et nombre de batch à passer
        if 'duree' in(transcriptions):
            nb_lignes_restantes = (pd.isna(transcriptions['duree'])).sum()
            
            # nb de batch total
            nb_batch = len(transcriptions['chapter_id'].value_counts())
            # première ligne vide sur durée
            pl = transcriptions['chapter_id'].loc[transcriptions['duree'].isna()].index[0]
            # on soustrait le nombre de batch distincts avant la première ligne vide sur durée
            nb_batch -= len(transcriptions[:pl]['chapter_id'].drop_duplicates())
            
        else:
            nb_lignes_restantes = len(transcriptions)
            nb_batch = len(transcriptions['chapter_id'].value_counts())
            
    else:
        # Nombre de ligne à traiter
        if 'duree' in(transcriptions):
            a = len(transcriptions['duree'])-(pd.isna(transcriptions['duree'])).sum()
            transcriptions =transcriptions['chapter_id'].drop_duplicates()
            
            b = 0
            for i in transcriptions.index:
                if i <= a:
                    b += 1
                else:
                    break
            transcriptions = transcriptions[b:b+nb_batch+1]
            nb_lignes_restantes = transcriptions[-1:].index[0]
            
            del a,b

        else:
            a = 0
            transcriptions =transcriptions['chapter_id'].drop_duplicates()
            transcriptions=transcriptions[:nb_batch+1]
            nb_lignes_restantes = transcriptions[-1:].index[0]
            
            del a
    
    del transcriptions
    print("Nombre de batch à traiter {}".format(nb_batch))
    print("Nombre de ligne à traiter {}".format(nb_lignes_restantes))
    print("===============================================\n\n")
    
    t0 = time()
    # Boucle de traitement de batch pour un nombre limité de batch
    for i in range(nb_batch):
        
        # Traitement du batch
        nb_lignes_batch, t1 = preprocess_batch(num_batch=i, fichier=fichier, sauvegarde = sauvegarde)
        
        # Calcul du nombre de ligne restante 
        nb_lignes_restantes -= nb_lignes_batch
        
        # Estimation du temps restant
        t_estime = seconde(t1/nb_lignes_batch*nb_lignes_restantes)
        
        print("\nNombre de batch fait {}/{}\n\n".format(i+1,nb_batch))
        print("\nNombre de lignes à traiter :", nb_lignes_restantes)
        print("Durée estimé restante :", t_estime)
        print("===============================================\n\n")
    
    # Heure de fin de la fonction
    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    print("Heure de fin :",date_time)
    
    t1 = int(time() - t0)
    print("Traitement Réalisé en {}".format(seconde(t1)))
    

def preprocess_batch(num_batch, fichier, sauvegarde = False):
    t0 = time()
    
    # ouverture du data frame
    transcriptions = pd.read_csv(fichier)
    
    # Controle de l'existance de la colonne duree
    if 'duree' in(transcriptions):
        # Identification de la première ligne vide et de l'id batch (chapter_id)
        a = len(transcriptions['duree'])-(pd.isna(transcriptions['duree'])).sum()
        id = transcriptions.loc[a,'chapter_id']
        
    else :
        #Initialisation de l'id batch (chapter_id)
        a=0
        id=transcriptions.loc[0,'chapter_id']
        # création de la nouvelle colonne vide
        transcriptions['duree']=np.nan
    
    # Définition du batch à traiter
    batch = transcriptions[transcriptions['chapter_id'] == id].reset_index()
    
    # Récupération et sauvegarde des audio_time
    audio_times = Recuperation_audio_times(batch['chemin'], sauvegarde = sauvegarde)
    
    # Récupération de la durée
    duree = Recuperation_duree(audio_times)
        
    # valorisation de n ligne de la colonne duree
    transcriptions.loc[a:a+len(duree)-1, 'duree']=duree
        
    # sauvegarde du dataframe à jour
    transcriptions.to_csv(fichier, index=False)
    
    # Affichage compte rendu
    print("Batch numéro {} terminé".format(num_batch+1))
    print("Id batch :",id)
    print("\nNombre de ligne traité :", len(batch))
    t1 = int(time() - t0)
    print("Réalisé en {}".format(seconde(t1)))
        
    return len(batch), t1


def seconde(nb_sec):
 
    heure = int((nb_sec / 3600))
 
    minute = int((nb_sec - (3600 * heure)) / 60)
 
    seconde = int(nb_sec - (3600 * heure) - (60 * minute))
 
    time = "{}:{}:{} (h:min:sec)".format(heure, minute, seconde)
 
    return time

""" Transformation de l'audio en image """

def spectrogram(audio,fft_length = 10000, affichage=False, save=False, fichier="spectrogram.JPG"):
    # Lien de référence pour modifier les paramètre d'un melspectrogram
    # https://librosa.org/doc/main/auto_examples/plot_display.html
    
    # définition d'un transformer (short time fournier transforme) 
    Stft = librosa.stft(audio, n_fft=512)

    # modification de l'amplitude en décibel pour la transfo de fournier du son y
    audio_db = librosa.amplitude_to_db(np.abs(Stft), ref=np.max)
    
    # permet d'avoir des valeurs plus compréhensible au niveau de l'échelle db
    audio_db = 80*np.ones_like(audio_db)+audio_db
   
    # Sauvegarde du spectrogramme
    if save:
        Sauvegarde_spectro_mfcc(audio_db, fichier)
    
    # Affichage du spectrogramme => implique qu'on ne récupère que l'image en retour
    if affichage:
        Affichage_spectro_mfcc(audio_db)

    return audio_db


def Mel_spectrogram(audio, sr=22050, affichage=False, save=False, fichier="out.JPG"):
    # Lien de référence pour modifier les paramètre d'un melspectrogram
    # https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html
    S = librosa.feature.melspectrogram(y=audio,
                                       sr=sr,
                                       n_mels=128 * 2,)
    
    # modification de l'amplitude en décibel pour la transfo de fournier du son y
    S_db_mel = -librosa.amplitude_to_db(S, ref=np.max)
   
    # Sauvegarde du Melspectrogramme
    if save:
        Sauvegarde_spectro_mfcc(S_db_mel, fichier)
    
    # Affichage du Melspectrogramme => implique qu'on ne récupère que l'image en retour
    if affichage:
        Affichage_spectro_mfcc(S_db_mel)
        
    return S_db_mel



def MFCC(audio, sr=22050, n_mfcc=40, affichage=False, save=False, fichier="out.JPG"):
    # Lien de référence pour modifier les paramètre d'un mfcc
    # https://librosa.org/doc/main/generated/librosa.feature.mfcc.html
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, hop_length=1024, htk=True, n_mfcc=40)
    
    # Sauvegarde du mfcc
    if save:
        Sauvegarde_spectro_mfcc(mfcc, fichier)
    
    # Affichage du mfcc => implique qu'on ne récupère que l'image en retour
    if affichage:
        Affichage_spectro_mfcc(mfcc)
        
    return mfcc


""" Affichage de l'image et/ou son """

def Affichage_spectro_mfcc(son):
    fig, ax = plt.subplots(figsize=(10, 5))
    img = librosa.display.specshow(son,
                                   sr=22050,
                                   hop_length  = 128,
                                   n_fft = 512,
                                   win_length = 512,
                                   x_axis='time',
                                   y_axis='log',
                                   ax=ax
                                   )
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    plt.show()
    
    
    
def Sauvegarde_spectro_mfcc(son, fichier="out.JPG"):
    fig, ax = plt.subplots(figsize=(10, 5))
    img = librosa.display.specshow(son,
                                   cmap = 'gray'
                                   )
    plt.axis("off")
    
    plt.savefig(fichier,dpi=1000, bbox_inches='tight',pad_inches = 0)
    
    
def Lecture_audio(audio, sr=22050):
    return ipd.Audio(audio, rate=sr)

""" Fonction pour récupérer la durée de l'audio si on utilise fonction tf au lieu de librosa"""
def Recuperation_duree_tf(files):
    duree = []
    for audi in files:
        audio = tfio.audio.AudioIOTensor(audi)
        duree.append(int(audio.shape[0])/int(audio.rate))
        
    return duree
        
""" Fonction de pre-process pour récupérer la durée de l'audio si on utilise fonction tf au lieu de librosa"""        
def Preprocess_simple(fichier='open_SLR_data/text_data.csv'):
    # ouverture du data frame
    transcriptions = pd.read_csv(fichier)
    
    # création de la nouvelle colonne vide
    transcriptions['duree']=np.nan
    
    # Récupération de la durée
    duree = Recuperation_duree_tf(transcriptions['chemin'])
    
    # valorisation de la colonne duree
    transcriptions['duree'] = duree
    
    # sauvegarde du dataframe à jour
    transcriptions.to_csv(fichier, index=False)
    
""" Séparation du dataframe en jeu de test, de validation et d'entraînement """
def split_dataframe(fichier='open_SLR_data/text_data.csv'):
    # ouverture du data frame
    df = pd.read_csv(fichier)
    
    # split
    df_train_val, df_test = train_test_split(df, test_size=0.10, random_state=42)
    df_train, df_val = train_test_split(df_train_val, test_size=0.10, random_state=42)

    # sauvegarde
    df_train.to_csv('open_SLR_data/df_train.csv', index=False)
    df_val.to_csv('open_SLR_data/df_val.csv', index=False)
    df_test.to_csv('open_SLR_data/df_test.csv', index=False)
    
    return print("Les datas splitées sont bien sauvegardées dans le dossier open_SLR_data")
    
""" Séparation du dataframe en jeu de test, de validation et d'entraînement """
def Lecture_data_split(f_train = 'open_SLR_data/df_train.csv',
                       f_val = 'open_SLR_data/df_val.csv', 
                       f_test = 'open_SLR_data/df_test.csv'):
    
    # ouverture des fichier
    df_train = pd.read_csv(f_train)
    df_val = pd.read_csv(f_val)
    df_test = pd.read_csv(f_test)
    
    return df_train, df_val, df_test