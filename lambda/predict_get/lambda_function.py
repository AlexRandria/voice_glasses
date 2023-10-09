import sys
import logging
import pymysql
import os
import json

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
    This function creates a new RDS database table 'predict'
    """
    # Liste les N dernières prédictions
    with conn.cursor() as cur:        
        sql = "SELECT * FROM predict ORDER BY timestamp DESC LIMIT %s" %(event['limit'] if 'limit' in event else 10)
        cur.execute(sql)
        output = cur.fetchall()
    conn.commit()

    print(output)

    return {
        'statusCode': 200,
        'result':json.dumps(output, default=str),
        'item_count': len(output)
    }