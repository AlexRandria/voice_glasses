import sys
import logging
import pymysql
import json
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
    sql_string = f"insert into test (id_audio, speaker_id, chapter_id, id_line, chemin, transcription) values('{id_audio}', '{Name}', '{Name}', '{Name}', '{Name}', '{Name}')"

    with conn.cursor() as cur:
        cur.execute("drop table test")
        cur.execute("create table if not exists test ( id_audio varchar(10) NOT NULL, speaker_id varchar(10) NOT NULL, chapter_id varchar(10) NOT NULL, id_line varchar(10) NOT NULL, chemin varchar(500) NOT NULL, transcription varchar(500) NOT NULL, PRIMARY KEY (id_audio))")
        cur.execute(sql_string)
        conn.commit()
        cur.execute("select * from test")
        output = cur.fetchall()
        logger.info("The following items have been added to the database:")
        logger.info("Test depuis github Actions 6")
        for row in cur:
            item_count += 1
            logger.info(row)
    conn.commit()

    return {
        'result':json.dumps(output),
        'statusCode': 200
    }