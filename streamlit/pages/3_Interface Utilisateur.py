import numpy as np
import pandas as pd
import streamlit as st
import librosa
import librosa.display
from io import BytesIO
import matplotlib.pyplot as plt
from libs.st_audiorec_component import st_audiorec

import requests
from requests.auth import HTTPBasicAuth
import boto3

aws_api_url = "https://ky9x9gezmj.execute-api.eu-west-3.amazonaws.com/v1"
aws_api_headers = {'Accept': 'application/json'}
aws_api_auth = HTTPBasicAuth('apikey', 'k8kLFxiyx7M9OcJXzCwi5fzaJkGYMaj1bm8zbVzf')
aws_access_key_id = "AKIAXRPAU26UJJQDJXQC"
aws_secret_access_key = "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"



url_dataset = 'd:/IA/sources/librispeech/train-clean-360'

def charger_audio(sDossier, iChapitre, iLigne, iLecteur, bGraphiques = True):

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

tab_param, tab_predict= st.tabs(["Paramétrage","Prédiction"])

# Paramétrage
with tab_param:
    st.write("")
    st.write("")

    st.write("")
    st.markdown("#### Seuil de confiance")

    response = requests.get(aws_api_url+"/param",
                            headers=aws_api_headers, auth=aws_api_auth)
    seuil = 0
    if response.ok:
        result = response.json()
    slider_seuil = st.slider("Seuil de confiance (%)",
                             min_value=0, max_value=100,
                             value=result["seuil_confiance"])

    st.write()
    if st.button("Valider"):
        response = requests.put(aws_api_url + "/param",
                                 json={"seuil_confiance":slider_seuil},
                                 headers=aws_api_headers, auth=aws_api_auth).json()
        if "statusCode" in response and response["statusCode"]==200:
            st.write("Nouveau paramètre enregistré !")
        else:
            st.write(response)
            st.write(response["errorMessage"])
            st.write(response["errorType"])
            st.write(response["stackTrace"])

# Prédictions
with tab_predict:

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

    bloc_prediction = False
    dataset = st.checkbox('Jeu de données Librispeech')
    record = st.checkbox('Effectuer un enregistrement')

    if dataset:

        col1, col2 = st.columns(2)

        # Livre
        with col1:
            sLivre = st.selectbox('Livr.', dfLivres['PROJECT TITLE'])
            iLivre = (dfLivres[dfLivres['PROJECT TITLE'] == sLivre]).index.values[0]

        # Chapitre
        with col2:
            dfChapitres = dfChapitres[dfChapitres['BOOK ID'] == iLivre]
            sChapitre = st.selectbox('Chap.', dfChapitres['CH. TITLE'])
            iChapitre = dfChapitres[(dfChapitres['CH. TITLE'] == sChapitre)].index.values[0]

        # Lecteur
        with col1:
            iLecteur = dfChapitres['READER'][iChapitre]
            dfLecteurs = dfLecteurs[dfLecteurs['ID'] == iLecteur]
            sLecteur = st.selectbox('Lect.', dfLecteurs['NAME'])

        with col2:
            sUrlFichier = "%s/%s/%s/%s-%s.trans.txt" % (url_dataset,
                                                        iLecteur,
                                                        iChapitre,
                                                        iLecteur,
                                                        iChapitre)
            dfLignes = pd.read_csv(sUrlFichier, delimiter='|', names=['col'])
            dfLignes['ID'] = dfLignes['col'].apply(lambda x: x.split(" ")[0])
            dfLignes['TXT'] = dfLignes['col'].apply(lambda x: x.replace(x.split(" ")[0], ''))
            sLigne = st.selectbox('Lign.', dfLignes['ID'])
            iLigne = dfLignes[dfLignes['ID'] == sLigne].index.values[0]

        "> " + dfLignes['TXT'][iLigne]

        signal, sr = charger_audio(sDossier=url_dataset,
                                   iChapitre=iChapitre,
                                   iLigne=iLigne,
                                   iLecteur=iLecteur,
                                   bGraphiques=False)

        col1, col2, col3 = st.columns(3)
        if col2.button('Prédiction'):
            bloc_prediction = True
            st.write("___")
            #st.write("> %s" % sUrlFichier)

    if record:
        wav_audio_data = st_audiorec()

        if wav_audio_data is not None:

            # st.audio(wav_audio_data, format='audio/wav')
            signal, sr = librosa.load(BytesIO(wav_audio_data))

            phrase = st.text_input(label="Phrase prononcée :", value="the members of the jury are really so nice to validate our diploma")

            col1, col2, col3 = st.columns(3)
            if col2.button('Prédiction'):
                bloc_prediction = True

                import soundfile as sf

                sf.write('sources/record.flac',
                         signal, int(sr),
                         format='flac'
                         )


                st.write("___")
                st.write("> Enregistrement")

    if bloc_prediction:

        if record:

            import uuid
            id = str(uuid.uuid4())

            s3bucket = "voiceglasses"
            key = "datasets/record/%s.flac" % id

            session = boto3.session.Session()
            s3 = session.client(service_name="s3",
                                region_name="eu-west-3",
                                aws_access_key_id=aws_access_key_id,
                                aws_secret_access_key=aws_secret_access_key)

            s3.upload_file(Filename='sources/record.flac',
                           Bucket='voiceglasses',
                           Key="datasets/record/%s.flac" % id)

            st.write("### Prédiction du modèle développé ")

            response = requests.get(aws_api_url + "/predict_recovoc",
                                    json={"id_record": id,
                                          "id_dataset": "",
                                          "transcription_real": phrase},
                                    headers=aws_api_headers, auth=aws_api_auth).json()

            print(">>> Réponse RecoVoc :")
            print(response)
            print(">>> ")

            if "statusCode" in response and response["statusCode"] == 200:
                st.write(response["transcription_predict"])
                st.write("##### CER : %s %%" % round(100 * response["cer"], 2))
                st.write("##### WER : %s %%" % round(100 * response["wer"], 2))
            else:
                st.write(response)

            st.write("### Prédiction du service Transcribe : ")

            response = requests.get(aws_api_url + "/predict_transcribe",
                                    json={"id_record" : id,
                                          "id_dataset" : "",
                                          "transcription_real" : phrase},
                                    headers=aws_api_headers, auth=aws_api_auth).json()

            print(">>> Réponse Transcribe :")
            print(response)
            print(">>> ")

            if "statusCode" in response and response["statusCode"] == 200:
                st.write(response["transcription_predict"])
                st.write("##### CER : %s %%" % round(100 * response["cer"], 2))
                st.write("##### WER : %s %%" % round(100 * response["wer"], 2))

            else:
                st.write(response)

        else :

            id = '%s-%s-%s' % (iLecteur,
                               iChapitre,
                               ('0000' + str(iLigne))[-4:])

            print(">>> %s" % id)
            if len(id) > 0 :

                st.write("### Prédiction du modèle développé ")

                response = requests.get(aws_api_url + "/predict_recovoc",
                                        json={"id_record": "",
                                              "id_dataset": id,
                                              "transcription_real": ""},
                                        headers=aws_api_headers, auth=aws_api_auth).json()

                print(">>> Réponse RecoVoc :")
                print(response)
                print(">>> ")

                if "statusCode" in response and response["statusCode"] == 200:
                    st.write(response["transcription_predict"])
                    st.write("##### CER : %s %%" % round(100*response["cer"],2))
                    st.write("##### WER : %s %%" % round(100*response["wer"],2))
                else:
                    st.write(response)

                st.write("### Prédiction du service Transcribe : ")
                response = requests.get(aws_api_url + "/predict_transcribe",
                                        json={"id_record": "",
                                              "id_dataset": id,
                                              "transcription_real": ""},
                                        headers=aws_api_headers, auth=aws_api_auth).json()

                print(">>> Réponse Transcribe :")
                print(response)
                print(">>> ")

                if "statusCode" in response and response["statusCode"] == 200:
                    st.write(response["transcription_predict"])
                    st.write("##### CER : %s %%" % round(100 * response["cer"], 2))
                    st.write("##### WER : %s %%" % round(100 * response["wer"], 2))
                else:
                    st.write(response)

            else:
                st.write("Veuillez sélectionner un enregistrement !")




