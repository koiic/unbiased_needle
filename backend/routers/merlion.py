import inspect
import json
import os
import re
import tempfile
from collections import OrderedDict
from enum import Enum
from datetime import datetime

import uuid

import boto3
from fastapi import APIRouter, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sagemaker.pytorch import PyTorch


merlion_router = APIRouter()


@merlion_router.get("/algorithms")
async def algoritms():
    return ["VAE", "LSTMED", "AutoEncoder"]


@merlion_router.get("/default_parameters/{algorithm_name}")
async def default_parameters(algorithm_name: str):
    assert algorithm_name in [
        "VAE",
        "LSTMED",
        "AutoEncoder",
    ], "Algorithm name not found"

    # model_class = ModelFactory.get_model_class(algorithm_name)

    # init_method = model_class.config_class.__init__

    if algorithm_name == "VAE":
        json_encoded_data = {
            "encoder_hidden_sizes": {"type": "<class 'tuple'>", "default": [25, 10, 5]},
            "decoder_hidden_sizes": {"type": "<class 'tuple'>", "default": [5, 10, 25]},
            "latent_size": {"type": "<class 'int'>", "default": 5},
            "sequence_len": {"type": "<class 'int'>", "default": 1},
            "kld_weight": {"type": "<class 'float'>", "default": 1},
            "dropout_rate": {"type": "<class 'float'>", "default": 0},
            "num_eval_samples": {"type": "<class 'int'>", "default": 10},
            "lr": {"type": "<class 'float'>", "default": 0.001},
            "batch_size": {"type": "<class 'int'>", "default": 1024},
            "num_epochs": {"type": "<class 'int'>", "default": 10},
            "max_score": {"type": "<class 'int'>", "default": 1000},
            "enable_calibrator": {"type": "<class 'bool'>", "default": True},
            "enable_threshold": {"type": "<class 'bool'>", "default": True},
        }
    elif algorithm_name == "LSTMED":
        json_encoded_data = {
            "hidden_size": {"type": "<class 'int'>", "default": 5},
            "sequence_len": {"type": "<class 'int'>", "default": 20},
            "n_layers": {"type": "<class 'tuple'>", "default": [1, 1]},
            "dropout": {"type": "<class 'tuple'>", "default": [0, 0]},
            "lr": {"type": "<class 'float'>", "default": 0.001},
            "batch_size": {"type": "<class 'int'>", "default": 256},
            "num_epochs": {"type": "<class 'int'>", "default": 10},
            "max_score": {"type": "<class 'int'>", "default": 1000},
            "enable_calibrator": {"type": "<class 'bool'>", "default": True},
            "enable_threshold": {"type": "<class 'bool'>", "default": True},
        }
    elif algorithm_name == "AutoEncoder":
        json_encoded_data = {
            "hidden_size": {"type": "<class 'int'>", "default": 5},
            "layer_sizes": {"type": "<class 'tuple'>", "default": [25, 10, 5]},
            "sequence_len": {"type": "<class 'int'>", "default": 1},
            "lr": {"type": "<class 'float'>", "default": 0.001},
            "batch_size": {"type": "<class 'int'>", "default": 512},
            "num_epochs": {"type": "<class 'int'>", "default": 50},
            "max_score": {"type": "<class 'int'>", "default": 1000},
            "enable_calibrator": {"type": "<class 'bool'>", "default": True},
            "enable_threshold": {"type": "<class 'bool'>", "default": True},
        }
    else:
        raise Exception("Algorithm not found")

    return JSONResponse(content=json_encoded_data)


@merlion_router.post("/algorithm/{algorithm_name}")
async def create_algorithm(
    algorithm_name: str, parameters: dict = Depends(default_parameters)
):
    if algorithm_name == "VAE":
        pass
        # model = ModelFactory.create("VAE", **parameters)
    elif algorithm_name == "LSTMED":
        pass
    elif algorithm_name == "AutoEncoder":
        pass
    else:
        raise Exception("Algorithm not found")

    return JSONResponse(parameters)


@merlion_router.post("/evaluate/{algorithm_name}")
async def create_model(
    algorithm_id: int,
    datasource_id: int,
):
    pass


@merlion_router.post("/train/{model_id}")
async def train_model(
    model_id: int,
    datasource_id: int,
    start_datetime: datetime,
    end_datetime: datetime,
    maio_token: str = None,
    train_test_split: float = 0.8,
):
    # Build model data file
    # create a tmp filename
    tmp_dirname = uuid.uuid4().hex

    model_data_path = os.getenv("MODEL_DATA_PATH", None)
    aws_role = os.getenv("AWS_ROLE", None)
    instance_type = os.getenv("INSTANCE_TYPE", None)

    print(f"aws_role: {aws_role}")
    print(f"instance_type: {instance_type}")
    print(f"model_data_path: {model_data_path}")

    # upload the model data to S3
    m = re.search("//(.+)/", model_data_path)
    if m:
        bucket_name = m.group(1)
    print(f"bucket_name: {bucket_name}")

    object_name = "code.tar.gz"

    # s3 = boto3.client("s3")

    # # status = s3.Bucket(bucket_name).upload_file(model_data_path, "model.tar.gz")
    # object_name = "model.tar.gz"
    # print(f"{bucket_name}/{tmp_dirname}")

    # s3.copy_object(
    #     CopySource=f"{bucket_name}/main/{object_name}",  # /Bucket-name/path/filename
    #     Bucket=bucket_name,  # Destination bucket
    #     Key=f"{tmp_dirname}/{object_name}",  # Destination path/filename
    # )

    source_dir = f"s3://{bucket_name}/code/{object_name}"  # because on S3
    output_path = f"s3://{bucket_name}/{tmp_dirname}/"

    # Create a PyTorch estimator

    # Get parameters from the algorithm
    algoritm_name = "LSTMED"

    print(f"source_dir: {source_dir}")
    print(f"output_path: {output_path}")

    hyperparameters = {
        "algorithm_name": algoritm_name,
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.01,
        "maio_instance_str": "heineken",
        "maio_token": maio_token,
        "datasource_id": datasource_id,
        "start_datetime": start_datetime.strftime("%Y-%m-%d_%H:%M:%S"),
        "end_datetime": end_datetime.strftime("%Y-%m-%d_%H:%M:%S"),
        "features": "CoolerTemp BathTemp CoolerSwitch RefridgentTemp CompressorCurrent",
        "train_test_split": train_test_split,
    }

    estimator = PyTorch(
        entry_point="script.py",
        source_dir="s3://maio-sagemaker/main/model.tar.gz",
        role=aws_role,
        instance_count=1,
        instance_type=instance_type,
        framework_version="2.0.0",
        py_version="py310",
        hyperparameters=hyperparameters,
        output_path=output_path,
    )

    estimator.fit(wait=False)

    return JSONResponse(
        {"status": True, "job_name": estimator.latest_training_job.job_name}
    )


@merlion_router.get("/train/{model_name}/status")
async def get_train_status(model_name: str):
    sagemaker_client = boto3.client("sagemaker")

    # Suppose 'your_training_job_name' is the name of your training job
    response = sagemaker_client.describe_training_job(TrainingJobName=model_name)

    # The response contains all the information about the training job
    return JSONResponse({"status": response["TrainingJobStatus"]})


@merlion_router.post("/predict/{model_id}")
def predict(model_id: int, data: dict):
    pass
