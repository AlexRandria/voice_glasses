import sys
import logging
import pymysql
import json
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

def lambda_handler(event, context):
    """
    This function creates a new RDS database table and writes records to it
    """
    if len(event['id']) == 0 or len(event['colonne']) == 0 or len(event['value']) == 0:
        return {
            'error': 'Error key missing',
            'id': event['id']
        }

    if event['value'] == 'TRUE':
        event['value'] = 1
    elif event['value'] == 'FALSE':
        event['value'] = 0

    with conn.cursor() as cur:
        sql = f"UPDATE dataset SET {event['colonne']} = '{event['value']}' WHERE id_audio = '{event['id']}'"
        print(sql)
        cur.execute(sql)
        sql = "SELECT * FROM dataset WHERE id_audio = '%s'" %(event['id'])
        print(sql)
        cur.execute(sql)
        output = cur.fetchall()
    conn.commit()

    return {
        'result':json.dumps(output),
        'statusCode': 200,
    }