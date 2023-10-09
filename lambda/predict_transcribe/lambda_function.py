import boto3
import json
import uuid
import os
import sys
import logging
import pymysql
import re
from jiwer import wer, cer

# rds settings
user_name = os.environ['USER_NAME']
password = os.environ['PASSWORD']
rds_host = os.environ['RDS_HOST']
db_name = os.environ['DB_NAME']

logger = logging.getLogger()
logger.setLevel(logging.INFO)

try:
    conn = pymysql.connect(host=rds_host, user=user_name, passwd=password, db=db_name, connect_timeout=5)
except pymysql.MySQLError as e:
    logger.error("ERROR: Unexpected error: Could not connect to MySQL instance.")
    logger.error(e)
    sys.exit()

logger.info("SUCCESS: Connection to RDS MySQL instance succeeded")

def lambda_handler(event, context):

    s3bucket = "voiceglasses"

    # Test sur le type de fichier son à transcrire
    if event['id_dataset']!="":
        id_audio = event['id_dataset']
        with conn.cursor() as cur:
            cur.execute("select * from dataset WHERE id_audio = '%s'"%(id_audio))
            output = cur.fetchone()
            chemin=output[4][20:].replace("/records","")
            s3object = "datasets/"+chemin
            transcription_real = output[5]
        conn.commit()
    else:
        id_audio = event['id_record']
        s3object = "datasets/record/" + id_audio + ".flac"
        transcription_real = event['transcription_real']

    filename = s3object.split("/")[-1]
    s3Path = "s3://" + s3bucket + "/" + s3object

    # Appel de l'application transcribe
    transcribe = boto3.client('transcribe')
    transcription_job_name = filename+ '-' + str(uuid.uuid4())
    transcription_key = "prediction/" + transcription_job_name + ".json"

    response = transcribe.start_transcription_job(
        TranscriptionJobName=transcription_job_name,
        LanguageCode='en-US',
        MediaFormat='flac',
        Media={
            'MediaFileUri': s3Path
        },
        OutputBucketName = "voiceglasses",
        OutputKey = transcription_key
    )

    # Attente de la fin du processus de transcription par transcribe
    status = transcribe.get_transcription_job(TranscriptionJobName=transcription_job_name)
    while status['TranscriptionJob']['TranscriptionJobStatus'] not in ['COMPLETED', 'FAILED']:
        status = transcribe.get_transcription_job(TranscriptionJobName=transcription_job_name)

    # Récupération du résultat de transcription sur S3
    s3 = boto3.client('s3')
    transcript_file = s3.get_object(Bucket='voiceglasses', Key=transcription_key)
    transcript_dict = json.loads(transcript_file['Body'].read().decode('utf-8'))
    transcription_predict = transcript_dict['results']['transcripts'][0]['transcript']
    
    # suppression de la ponctuation dans la prédiction et mise en majuscule
    transcription_predict = re.sub("[,;.:!?]","",transcription_predict).upper()
    
    w_error = wer(transcription_real, transcription_predict)
    c_error = cer(transcription_real, transcription_predict)
    
    """
    # Sauvegarde en base du lien de l'enregistrement et de la phrase réelle si le CER est supérieur au seuil de confiance
    with conn.cursor() as cur:
        cur.execute("select * from param")
        output = cur.fetchone()
        seuil_confiance=output[0]/100
    conn.commit()
    
    if (id_audio[:6] == "record") & (c_error > seuil_confiance) & (transcription_real!=""):
        with conn.cursor() as cur:
            cur.execute("create table if not exists dataset_record ( id_audio varchar(50) NOT NULL, transcription varchar(500) NOT NULL, flag_train boolean, PRIMARY KEY (id_audio))")
            cur.execute(f"insert into dataset_record values(%s,%s,%s)",(id_audio, transcription_real, False))
        conn.commit()
    """

    return {
        'statusCode': 200,
        'transcription_real': transcription_real,
        'transcription_predict': transcription_predict,
        'cer': c_error,
        'wer': w_error
        }
