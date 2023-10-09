# Reconnaissance vocale
Formation Datascientest
Cursus ML Ops
Promotion mai 2023

## Contexte : L'intelligence artificielle au service du handicap
L'objectif est de pouvoir retranscrire les paroles d'un interlocuteur sur des lunettes connectées, pour aider les personnes malentendantes.

Un premier modèle a été développé dans le cadre du cursus Data Science (promo d'octobre 2022).
Il s'agissait d'industrialiser ce modèle.


## Modélisation

### Extraction de caractéristiques

La parole est un enchevêtrement complexe de sons qu’il est impossible de décoder en l’état. Et la quantité d’information est telle qu’il serait très coûteux et peu performant d'entraîner un modèle directement sur le son brut. 
Il est nécessaire de convertir ces sons, de les synthétiser.

Ci-après un accord de piano (signal rouge en bas), composé des 3 notes représentées individuellement au-dessus : on comprend la difficulté de décoder l'accord simplement par son signal.
![image](https://github.com/Marc-ALLAIN-Verlingue/mai23_mlops_voice_glasses/assets/94903015/1a5dcc8d-37d7-4e2a-b9c7-6577b41f3447)

L’outil mathématique le plus utilisé est la transformée de Fourier, permettant une  décomposition fréquentielle du signal.
![image](https://github.com/Marc-ALLAIN-Verlingue/mai23_mlops_voice_glasses/assets/94903015/3cf8f78b-39fb-48d5-8c25-aad2353562bf)

La représentation la plus courante est le spectrogramme : une représentation fréquentielle d’un signal dans le temps
![image](https://github.com/Marc-ALLAIN-Verlingue/mai23_mlops_voice_glasses/assets/94903015/abc3170b-ea78-4a3e-839e-57833409144f)

### Le modèle

La méthode utilisée est donc d’appliquer une Transformée de Fourier courte (à fenêtre glissante), afin d'échantillonner le signal. 
On traite ainsi une succession de spectrogrammes représentant des signaux d‘environ 30 millisecondes.

Parmi les nombreux modèles qui existent, le modèle retenu a été le “Deepspeech2” (https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/deepspeech2.html)

![image](https://github.com/Marc-ALLAIN-Verlingue/mai23_mlops_voice_glasses/assets/94903015/a6b39e73-2cc6-4e4c-ba30-18d46e507b05)

Le modèle se décompose en trois grands parties :

➔	La partie convolutive avec deux couches « Conv2D » qui vont permettre de découper le spectrogramme en entrée en colonnes. 
On ajoute après chacune de ces couches une normalisation avec des couches « BatchNormalization » et « ReLU ». 
Cette partie va permettre de distinguer des « unités temporelles » que l'on pourra classifier comme un caractère spécifique.

➔	La partie récurrente en cinq couches de neurones récurrentes « GRU », avec l'utilisation de couche « Bidirectional » pour avoir une concaténation de l'entrée dans le sens normal et inversé. 
Cette partie va tenir compte de ce qu'il y a « avant » et « après » afin de retrouver un lien qui orientera notre classification.

➔	La partie “classique” avec deux couches « Dense » associé avec une couche « Relu » et « Dropout », pour faire la classification finale qui donne le résultat en sortie de la dernière couche.

### Fonction de perte

La fonction de perte est la CTC ou « Classification Temporelle Connectionniste ». 
Il s’agit d’un moyen de se retrouver dans une séquence temporelle sans connaître le lien entre l'entrée et la sortie.

### Métriques

Concernant les métriques, 2 ont été retenues et utilisées pour mesurer les performances du modèle : 

➔	Le CER : Character Error Rate, ou taux d’erreur sur les caractères

 ![image](https://github.com/Marc-ALLAIN-Verlingue/mai23_mlops_voice_glasses/assets/94903015/32ee3e72-a003-4f81-8587-7f03e9c23b01)


➔	Le WER : Word Error Rate, ou taux d’erreur sur les mots

 ![image](https://github.com/Marc-ALLAIN-Verlingue/mai23_mlops_voice_glasses/assets/94903015/d5d86fac-c426-4ebc-8d4b-eb9c7b36bd70)


où 
- S = nombre de Substitutions
- D = nombre de suppressions (Delete)
- I = nombre d’Insertions
- C = nombre de caractères/mots Corrects
- N = nombre de caractères/mots de référence = S + C + D

### Résultats & Performances

![image](https://github.com/Marc-ALLAIN-Verlingue/mai23_mlops_voice_glasses/assets/94903015/3017883d-ff37-4f39-8c91-d34045582f53)

## Architecture cible

![image](https://github.com/Marc-ALLAIN-Verlingue/mai23_mlops_voice_glasses/assets/94903015/cf97e252-06bb-4e78-a655-e73abb18c33c)


## Procédure d'installation

N'étant pas équipés de lunettes connectées, nous avons simulé l'interface avec une application Streamlit.
Pour installer l'application, il vous faut :
- Télécharger le dossier "streamlit" contenant l'intégralité de l'application
- Télécharger le dataset "train-clean-360" de LibriSpeech (https://www.openslr.org/) et le décomprésser dans le dossier "d:/IA/sources"
- créer un environnement Python 3.9 et y installer les requirements du fichier "requirements.txt" à l racine du dossier Streamlit
- lancer la commande streamlit run .\Introduction.p
