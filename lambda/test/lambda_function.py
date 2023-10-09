import json
import requests
import boto3
import uuid
from requests.auth import HTTPBasicAuth

aws_api_url = "https://ky9x9gezmj.execute-api.eu-west-3.amazonaws.com/v1"
aws_api_headers = {'Accept': 'application/json'}
aws_api_auth = HTTPBasicAuth('apikey', 'k8kLFxiyx7M9OcJXzCwi5fzaJkGYMaj1bm8zbVzf')
aws_access_key_id = "AKIAXRPAU26UJJQDJXQC"
aws_secret_access_key = "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"

def lambda_handler(event, context):
    # param_get
    response = requests.get(aws_api_url+"/param",headers=aws_api_headers, auth=aws_api_auth)
    if response.ok:
        result = response.json()
        response_param_get = result['statusCode']
    else:
        response_param_get="ko"
    print("response_param_get :",response_param_get)

    # param_put
    response = requests.put(aws_api_url + "/param",
                                json={"seuil_confiance":50},
                                headers=aws_api_headers, auth=aws_api_auth)
    if response.ok:
        result = response.json()
        response_param_put = result['statusCode']
    else:
        response_param_put="ko"
    print("response_param_put :",response_param_put)

    # predict_transcribe_record
    response = requests.get(aws_api_url + "/predict_transcribe",
                                json={"id_dataset": "",
                                    "id_record": "record-6c513421-2271-4abd-9178-91e9bf63027d",
                                    "transcription_real": "IN SIXTEEN SIXTY FIVE WRITTEN BY A CITIZEN WHO CONTINUED ALL THE WHILE IN LONDON NEVER MADE PUBLIC BEFORE"},
                                headers=aws_api_headers, auth=aws_api_auth)
    if response.ok:
        result = response.json()
        response_predict_transcribe_record = result['statusCode']
    else:
        response_predict_transcribe_record="ko"
    print("response_predict_transcribe_record :",response_predict_transcribe_record)

    # predict_transcribe_dataset
    response = requests.get(aws_api_url + "/predict_transcribe",
                                json={"id_dataset": "100-121669-0000",
                                    "id_record": "",
                                    "transcription_real": ""},
                                headers=aws_api_headers, auth=aws_api_auth)
    if response.ok:
        result = response.json()
        response_predict_transcribe_dataset = result['statusCode']
    else:
        response_predict_transcribe_dataset="ko"
    print("response_predict_transcribe_dataset :",response_predict_transcribe_dataset)

    # predict_recovoc_record
    s3bucket = "voiceglasses"
    session = boto3.session.Session()

    s3 = session.client(service_name="s3",
                        region_name="eu-west-3",
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key)

    id = str(uuid.uuid4())

    s3.copy_object(Bucket=s3bucket, Key="datasets/record/%s.flac" % id,
                CopySource={'Bucket':s3bucket, 'Key':'datasets/record/record-6c513421-2271-4abd-9178-91e9bf63027d.flac'})

    response = requests.get(aws_api_url + "/predict_recovoc",
                                json={"id_dataset": "",
                                    "id_record": id,
                                    "transcription_real": "IN SIXTEEN SIXTY FIVE WRITTEN BY A CITIZEN WHO CONTINUED ALL THE WHILE IN LONDON NEVER MADE PUBLIC BEFORE"},
                                headers=aws_api_headers, auth=aws_api_auth)
    if response.ok:
        result = response.json()
        response_predict_recovoc_record = result['statusCode']
    else:
        response_predict_recovoc_record="ko"
    print("response_predict_recovoc_record :",response_predict_recovoc_record)

    # predict_recovoc_dataset
    response = requests.get(aws_api_url + "/predict_recovoc",
                                json={"id_dataset": "100-121669-0000",
                                    "id_record": "",
                                    "transcription_real": ""},
                                headers=aws_api_headers, auth=aws_api_auth)
    if response.ok:
        result = response.json()
        response_predict_recovoc_dataset = result['statusCode']
    else:
        response_predict_recovoc_dataset="ko"
    print("response_predict_recovoc_dataset :",response_predict_recovoc_dataset)

    # dataset_record
    response = requests.get(aws_api_url + "/dataset_record",
                                json={},
                                headers=aws_api_headers, auth=aws_api_auth)
    if response.ok:
        result = response.json()
        response_dataset_record = result['statusCode']
    else:
        response_dataset_record="ko"
    print("response_dataset_record :",response_dataset_record)

    return {
        'statusCode': 200,
        'response_param_get': response_param_get,
        'response_param_put': response_param_put,
        'response_predict_transcribe_record': response_predict_transcribe_record,
        'response_predict_transcribe_dataset': response_predict_transcribe_dataset,
        'response_predict_recovoc_record': response_predict_recovoc_record,
        'response_predict_recovoc_dataset': response_predict_recovoc_dataset,
        'response_dataset_record': response_dataset_record
    }
