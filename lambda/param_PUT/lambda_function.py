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
    MAJ seuil_confiance
    """

    if type(event['seuil_confiance'])!=int or event['seuil_confiance'] < 0 or event['seuil_confiance'] > 100:
        return {
            'error': 'Error seuil_confiance value',
        }

    with conn.cursor() as cur:
        cur.execute(f"UPDATE param SET seuil_confiance={event['seuil_confiance']}")
    conn.commit()

    return {
        'statusCode': 200,
    }
