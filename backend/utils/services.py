import os

# call a lambda function url and pass in a payload

import requests
import json

from dotenv import load_dotenv

load_dotenv()

function_url = os.getenv("LAMBDA_FUNCTION_URL", None)

if function_url is None:
    raise Exception("LAMBDA_FUNCTION_URL is not set")


def call_lambda_function(payload: dict):
    # include payload as query strings
    try:
        url = f"{function_url}/prediction?data_source_name={payload['gateway_name']}&endpoint={payload['endpoint']}&start_time={payload['start_time']}&token={payload['token']}"
        response = requests.get(url)
        return json.loads(response.text)

    except Exception as e:
        print(e)
        return None
