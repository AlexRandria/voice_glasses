import numpy as np
import pandas as pd
import streamlit as st
import librosa
import librosa.display
from io import BytesIO
import matplotlib.pyplot as plt
from libs.st_audiorec_component import st_audiorec
from PIL import Image

import tensorflow as tf
import tensorflow_io as tfio
from libs.Module_reco_voc_2 import Fabriquer_ou_restorer_model
import keras
from torchmetrics.text import CharErrorRate, WordErrorRate

url_dataset = 'd:/IA/sources/librispeech/train-clean-360'


def charger_audio(sDossier, iChapitre, iLigne, iLecteur, bGraphiques=True):
    "Fichier  '%s-%s-%s.flac'" % (iLecteur, iChapitre, ('0000' + str(iLigne))[-4:])
    ficAudio = '%s/%s/%s/%s-%s-%s.flac' % (sDossier,
                                           iLecteur,
                                           iChapitre,
                                           iLecteur,
                                           iChapitre,
                                           ('0000' + str(iLigne))[-4:])

    st.audio(ficAudio)
    y, sr = librosa.load(ficAudio)

    if bGraphiques:
        "##### a) Enveloppe globale"
        fig, ax = plt.subplots()
        ax.plot(pd.Series(y))
        st.pyplot(fig)

        "##### b) Décomposition Harmonique/Percusive"
        fig, ax = plt.subplots()
        y_harm, y_perc = librosa.effects.hpss(y)
        ax.plot(y_harm, alpha=0.5, label='Harmonic')
        ax.plot(y_perc, alpha=0.5, color='r', label='Percussive')
        ax.legend()
        st.pyplot(fig)

        "##### c) Transformée de Fourier"
        FRAME_SIZE = 2048
        HOP_SIZE = 512
        y_stft = librosa.stft(y, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
        y_db = librosa.power_to_db(np.abs(y_stft) ** 2)

        # Plot the signal read from wav file
        fig = plt.figure()
        plt.subplot(211)
        plt.title(f"Spectrogram")

        plt.plot(y)
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")

        plt.subplot(212)
        plt.specgram(y, Fs=sr)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        st.pyplot(fig)

    return y, sr


tab_theorie, tab_data, tab_prep, tab_model = st.tabs(["Théorie",
                                                       "Jeu de données",
                                                       "Préparation",
                                                       "Modèle"])

########################################################################################################################
# > THEORIE
########################################################################################################################

with tab_theorie:
    st.write("")
    st.write("")

    st.markdown("## Transformée de Fourier")

    """Le principe de la transformé de Fourier est de décomposer une fonction non périodique 
    en plusieurs composantes périodiques. Prenons l'exemple d'un accord de piano : cet accord 
    est la combinaison de plusieurs notes distinctes, et le résultat est un son spécifique. 
    Si on regarde la représentation temporelle de ce son on aura une fonction non périodique. 
    Mais si on applique la transformé de Fourier on retrouvera nos différentes notes 
    et leur fonctions périodiques."""

    st.image("images/son_piano.png")

    st.image("images/son_fourier.png")

    st.markdown("## Spectrogramme")

    """L'objectif est d'avoir une représentation de l'**empreinte fréquentielle** du son
    en fonction du temps. Ce graphique, le spectrogramme, se présente comme ceci :"""

    st.image("images/son_spectrogramme.png")

    """Les zones de couleur plus claires indiquent une plus grande 
    intensité de fréquence, tandis que les zones plus sombres indiquent une intensité plus 
    faible."""

    st.markdown("## FFT : Fast Fourier Transformation")

    """Le principe de la transformation de Fourier courte (ou transformation de Fourier à 
    fenêtre glissante) est d'appliquer à un tronçon local le principe de la transformation 
    de Fourier. Dans notre cas, cela revient à dire que sur notre audio d'une durée de 
    [x secondes] on ne va tenir compte que de ce qui se passe sur une fenêtre de 0,3 seconde 
    (exemple). Chaque tronçon est traité indépendamment des autres. Ainsi on va pouvoir appliquer 
     la méthode au même rythme que le locuteur qui va moduler l'état de son corps vocal pour 
     produire des sons variés."""

    st.image("images/son_fft.png")

    """Ainsi, notre problématique de reconnaissance vocale se ramène à une problématique 
     de traitement de séquences d'images."""

########################################################################################################################
# > JEU DE DONNEES
########################################################################################################################

with tab_data:
    st.write("")
    st.write("")
    st.markdown("## Librispeech")

    # Jeux de données")

    "### 1. Présentation"

    "> OpenSLR est un site dédié à l'hébergement de ressources vocales et linguistiques, telles que des corpus d'entrainement de modèles de reconnaissance vocale 'Librispeech'"

    "https://www.openslr.org/12/"

    dfFichiers = pd.DataFrame([['train-clean-100.tar.gz', 28_288, '6.5 Go', '~100 h.'],
                               ['train-clean-360.tar.gz', '***', '23 Go', '~360 h.'],
                               ['train-other-500.tar.gz', '***', '30 Go', '~500 h.']],
                              columns=['archive', 'nb fichiers', 'taille', 'durée totale'])
    dfFichiers

    "### 2. Echantilllons"

    # Infos Chapitres
    dfChapitres = pd.read_csv("%s/CHAPTERS.TXT" % url_dataset, delimiter='|')
    dfChapitres.rename(columns={"ID    ": "ID",
                                ' PROJECT TITLE': 'PROJECT TITLE',
                                ' CH. TITLE ': 'CH. TITLE',
                                ' PROJ.': 'PROJ.',
                                ' SUBSET           ': 'SUBSET'}, inplace=True)
    dfChapitres = dfChapitres[dfChapitres['SUBSET'] == ' train-clean-360  ']
    dfChapitres.set_index('ID', inplace=True)

    # Infos Livres
    dfLivres = (dfChapitres.groupby(by=['BOOK ID', 'PROJECT TITLE'], as_index=False, ).count())[
        ['BOOK ID', 'PROJECT TITLE']]
    dfLivres.drop_duplicates(subset=['BOOK ID'], keep='first', inplace=True)
    dfLivres.set_index('BOOK ID', inplace=True)

    # Infos lecteur
    dfLecteurs = pd.read_csv("%s/SPEAKERS.TXT" % url_dataset, delimiter='|')
    dfLecteurs.rename(columns={"ID  ": "ID",
                               " NAME": "NAME"}, inplace=True)
    # dfLecteurs.set_index('ID', inplace=True)

    col1, col2 = st.columns(2)

    # Livre
    with col1:
        sLivre = st.selectbox('Livre', dfLivres['PROJECT TITLE'])
        iLivre = (dfLivres[dfLivres['PROJECT TITLE'] == sLivre]).index.values[0]

    # Chapitre
    with col2:
        dfChapitres = dfChapitres[dfChapitres['BOOK ID'] == iLivre]
        sChapitre = st.selectbox('Chapitre', dfChapitres['CH. TITLE'])
        iChapitre = dfChapitres[(dfChapitres['CH. TITLE'] == sChapitre)].index.values[0]

    # Lecteur
    with col1:
        iLecteur = dfChapitres['READER'][iChapitre]
        dfLecteurs = dfLecteurs[dfLecteurs['ID'] == iLecteur]
        sLecteur = st.selectbox('Lecteur', dfLecteurs['NAME'])

    with col2:
        sUrlFichier = "%s/%s/%s/%s-%s.trans.txt" % (url_dataset,
                                                    iLecteur,
                                                    iChapitre,
                                                    iLecteur,
                                                    iChapitre)
        dfLignes = pd.read_csv(sUrlFichier, delimiter='|', names=['col'])
        dfLignes['ID'] = dfLignes['col'].apply(lambda x: x.split(" ")[0])
        dfLignes['TXT'] = dfLignes['col'].apply(lambda x: x.replace(x.split(" ")[0], ''))
        sLigne = st.selectbox('Ligne', dfLignes['ID'])
        iLigne = dfLignes[dfLignes['ID'] == sLigne].index.values[0]

    "> " + dfLignes['TXT'][iLigne]

    charger_audio(sDossier=url_dataset,
                  iChapitre=iChapitre,
                  iLigne=iLigne,
                  iLecteur=iLecteur)

    "### 3. Analyse du jeu de données"

    """Le jeu de données utilisé est le train-clean-100, qui correspond à 100 heures d'audio en anglais.
- Nombre de speakers : 251
- Nombre de chapitres audio : 5831
- Nombre de morceaux audio : 28539
    """

    st.markdown("#### Répartition par genre")

    """Une répartition équibibrée des genres :"""
    st.image("images/dataset_genre.png")

    st.markdown("#### Répartition par speaker & chapitre")
    """Une répartition équitable du nombre et de la durée des enregistrements entre les chapitres 
    et les speakers : """
    col_duree, col_nb = st.columns(2)

    with col_duree:
        st.image("images/dataset_duree_speaker.png")
        st.image("images/dataset_duree_chapitre.png")

    with col_nb:
        st.image("images/dataset_nb_records_speaker.png")
        st.image("images/dataset_nb_records_chapitre.png")

    st.markdown("#### Durée des enregistrements")
    """Répartition des durées des enregistrements :"""
    st.image("images/dataset_duree.png")

########################################################################################################################
# > DATA PREPARATION
########################################################################################################################

with tab_prep:
    st.write("")
    st.write("")
    st.markdown("## Préparation des données")

    """
    Faute de temps et nous basant sur des jeux de données de très bonne qualité, nous n'avons effectué 
    aucun préprocessing, si ce n'est la génération des spectrogrammes via la librairie TensorFlow.

    \n\n
    Différents traitements pourraient améliorer le modèle :
        \n- supprimer les blancs en début et fin d'enregistrement
        - ajouter du bruit aléatoirement sur certains enregistrements pour rendre le modèle plus robuste
        - ...
    """

########################################################################################################################
# > MODELISATION
########################################################################################################################

with tab_model:
    st.write("")
    st.write("")
    st.markdown("## Modélisation")

    st.markdown("### CTC ou « Classification Temporelle Connectionniste »")
    """
    Il s'agit d'un moyen de se retrouver dans une séquence temporelle sans connaître le lien entre l'entrée et la sortie.

    \n\nElle calcule la probabilité au niveau de chaque caractère d'un chemin. 
    Elle en fait le produit, puis elle tient compte du symbole blanc (epsilon) 
    ainsi que des chemins qui produisent la même séquence, pour enfin sélectionner 
    la séquence avec la plus grande probabilité.
    """

    st.image(Image.open('images/ctc.png'))
    st.image(Image.open('images/ctc.jpg'))

    st.markdown("### Choix du modèle")
    """
    Un modèle un peu plus complexe : nous avons sélectionné celui Deepspeech2 de Baidu, basé sur TensorFlow (https://github.com/noahchalifour/baidu-deepspeech2)
    """
    st.image(Image.open('images/recovoc_archi.jpg'))

    st.markdown("### Métriques d'apprentissage")

    st.image("images/recovoc_metrics.jpg")

