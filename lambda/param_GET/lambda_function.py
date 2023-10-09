import sys
import logging
import pymysql
import os


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
    """
    This function creates a new RDS database table 'param' and read seuil_confiance record to it
    """
    with conn.cursor() as cur:
        cur.execute("create table if not exists param ( seuil_confiance int NOT NULL, PRIMARY KEY (seuil_confiance))")
        cur.execute("select * from param")
        output = cur.fetchone()

        if output is None:
            cur.execute("insert into param (seuil_confiance) values ('60')")
            seuil_confiance=60
        else:
            seuil_confiance=output[0]
    conn.commit()

    return {
        'seuil_confiance':seuil_confiance,
        'statusCode':200
    }