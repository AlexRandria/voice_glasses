import sys
import logging
import pymysql
import csv
import os

from datetime import datetime

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

def lambda_handler(event, context):
    """
    This function creates a new RDS database table and writes records to it
    """
    Name = 'test'
    now = datetime.now()
    id_audio = now.strftime("%d%H%M%S")
    item_count = 0

    import boto3

    s3 = boto3.client('s3')
    csv_file_obj = s3.get_object(Bucket='voiceglassestest', Key='DATASET.csv')
    csv_data = csv_file_obj['Body'].read().decode('utf-8').splitlines()

    results = []
    for row in csv.reader(csv_data):
        if item_count == 0:
            item_count += 1
            continue
        results.append(row)

    sql_string = f"insert into dataset (id_audio, speaker_id, chapter_id, id_line, chemin, transcription) values(%s,%s,%s,%s,%s,%s)"

    with conn.cursor() as cur:
        #cur.execute("drop table dataset")
        cur.execute("create table if not exists dataset ( id_audio varchar(20) NOT NULL, speaker_id varchar(10) NOT NULL, chapter_id varchar(10) NOT NULL, id_line varchar(10) NOT NULL, chemin varchar(500) NOT NULL, transcription varchar(500) NOT NULL, flag_train boolean, PRIMARY KEY (id_audio))")
        for row in results:
            cur.execute(sql_string, tuple(row))
            conn.commit()
        cur.execute("select count(id_audio) from dataset")
        output = cur.fetchall()
        logger.info("The following items have been added to the database:")
        for row in cur:
            item_count += 1
            logger.info(row)
    conn.commit()

    return {
        'result':len(output),
        'statusCode': 200
    }