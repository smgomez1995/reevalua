import boto3
import json
import pandas as pd

# Define the SageMaker client
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")
endpoint_name = "reevalua-prueba-tecnica-2025-01-02-19-17-28"


def call_endpoint(input_data):
    processed_input_data = json.loads(json.dumps(input_data, default=lambda x: None))
    payload = json.dumps(processed_input_data)
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload,
    )
    return json.loads(response["Body"].read().decode("utf-8"))
