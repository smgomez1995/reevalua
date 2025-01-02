import boto3
from sagemaker.sklearn import SKLearnModel
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import CSVSerializer, JSONSerializer
from datetime import datetime

s3 = boto3.client("s3")
bucket_name = "reevalua-trial"
s3.upload_file("model.tar.gz", bucket_name, "voting_classifier/model.tar.gz")

def deploy_model_in_endpoint():
    role = "arn:aws:iam::590183791256:role/service-role/AmazonSageMaker-ExecutionRole-20241223T184677"
    dependencies = ["requirements.txt"]
    endpoint_name = "reevalua-prueba-tecnica-" + str(datetime.now()).replace(" ", "-").replace(":", "-").split(".")[0]

    # Define the model
    model = SKLearnModel(
        model_data=f"s3://{bucket_name}/voting_classifier/model.tar.gz",
        role=role,
        framework_version="1.2-1",
        entry_point="model.py",
        dependencies=dependencies
    )

    # Deploy the model to an endpoint
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
        endpoint_name=endpoint_name
    )
    print(f"Deployed Endpoint: {endpoint_name}")
    return predictor