import inspect
import json
import os
import re
import tempfile
import uuid
from collections import OrderedDict
from datetime import datetime
from enum import Enum
from typing import List

import boto3
from database import engine
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from models.ml_model import (
    Model,
    ModelCreate,
    ModelRead,
    ModelUpdate,
    ModelVersion,
    ModelVersionCreate,
    ModelVersionRead,
    ModelVersionUpdate,
    ModelVersionStatus,
)
from sagemaker.pytorch import PyTorch, PyTorchModel
from sqlmodel import Session, select

merlion_router = APIRouter()


@merlion_router.get("/default_parameters/{algorithm_name}")
async def default_parameters(algorithm_name: str):
    assert algorithm_name in [
        "VAE",
        "LSTMED",
        "AutoEncoder",
    ], "Algorithm name not found"

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


@merlion_router.post("/models/", response_model=ModelRead)
def create_model(model: ModelCreate):
    with Session(engine) as session:
        db_model = Model.from_orm(model)
        session.add(db_model)
        session.commit()
        session.refresh(db_model)
        return db_model


@merlion_router.get("/models/", response_model=List[ModelRead])
def read_models(offset: int = 0, limit: int = Query(default=100, lte=100)):
    with Session(engine) as session:
        modeles = session.exec(select(Model).offset(offset).limit(limit)).all()
        return modeles


@merlion_router.get("/models/{model_id}", response_model=ModelRead)
def read_model(model_id: int):
    with Session(engine) as session:
        model = session.get(Model, model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        return model


@merlion_router.patch("/models/{model_id}", response_model=ModelRead)
def update_model(model_id: int, model: ModelUpdate):
    with Session(engine) as session:
        db_model = session.get(Model, model_id)
        if not db_model:
            raise HTTPException(status_code=404, detail="Model not found")
        model_data = model.dict(exclude_unset=True)
        for key, value in model_data.items():
            setattr(db_model, key, value)
        session.add(db_model)
        session.commit()
        session.refresh(db_model)
        return db_model


@merlion_router.post("/model_versions/", response_model=ModelVersionRead)
def create_model_versions(model_version: ModelVersionCreate):
    with Session(engine) as session:
        db_model_version = ModelVersion.from_orm(model_version)
        session.add(db_model_version)
        session.commit()
        session.refresh(db_model_version)
        return db_model_version


@merlion_router.get("/model_versions/", response_model=List[ModelVersionRead])
def read_model_versions(offset: int = 0, limit: int = Query(default=100, lte=100)):
    with Session(engine) as session:
        model_versions = session.exec(
            select(ModelVersion).offset(offset).limit(limit)
        ).all()
        return model_versions


@merlion_router.get(
    "/model_versions/{model_version_id}", response_model=ModelVersionRead
)
def read_model_version(model_version_id: int):
    with Session(engine) as session:
        model_version = session.get(ModelVersion, model_version_id)
        if not model_version:
            raise HTTPException(status_code=404, detail="ModelVersion not found")
        return model_version


@merlion_router.patch(
    "/model_versions/{model_version_id}", response_model=ModelVersionRead
)
def update_model_version(model_version_id: int, model_version: ModelVersionUpdate):
    with Session(engine) as session:
        db_model_version = session.get(ModelVersion, model_version_id)
        if not db_model_version:
            raise HTTPException(status_code=404, detail="ModelVersion not found")
        model_version_data = model_version.dict(exclude_unset=True)
        for key, value in model_version_data.items():
            print(f"key: {key}, value: {value}")
            setattr(db_model_version, key, value)
        session.add(db_model_version)
        session.commit()
        session.refresh(db_model_version)
        return db_model_version


@merlion_router.post(
    "/model_versions/{model_version_id}/train", response_model=ModelVersionRead
)
async def train_model_version(model_version_id: int):
    with Session(engine) as session:
        db_model_version = session.get(ModelVersion, model_version_id)
        if not db_model_version:
            raise HTTPException(status_code=404, detail="ModelVersion not found")

        if db_model_version.job_name is not None:
            raise HTTPException(status_code=404, detail="ModelVersion already trained")

        db_model = session.get(Model, db_model_version.model_id)
        if not db_model:
            raise HTTPException(status_code=404, detail="Model not found")

        keys = db_model_version.algorithm_parameters.keys()

        hyperparameters = dict()
        hyperparameters["algorithm_name"] = db_model_version.algorithm_name
        if "maio_instance_str" not in keys:
            raise HTTPException(status_code=404, detail="maio_instance_str not found")
        hyperparameters["maio_instance_str"] = db_model_version.algorithm_parameters[
            "maio_instance_str"
        ]
        if "maio_token" not in keys:
            raise HTTPException(status_code=404, detail="maio_token not found")
        hyperparameters["maio_token"] = db_model_version.algorithm_parameters[
            "maio_token"
        ]
        hyperparameters["datasource_id"] = db_model_version.datasource_id
        hyperparameters["start_datetime"] = db_model_version.start_datetime.strftime(
            "%Y-%m-%d_%H:%M:%S"
        )
        hyperparameters["end_datetime"] = db_model_version.end_datetime.strftime(
            "%Y-%m-%d_%H:%M:%S"
        )
        hyperparameters["features"] = " ".join(db_model.tag_names)
        hyperparameters["train_test_split"] = db_model_version.train_test_split

        model_data_path = os.getenv("MODEL_DATA_PATH", None)
        aws_role = os.getenv("AWS_ROLE", None)
        instance_type = os.getenv("INSTANCE_TYPE", None)

        # upload the model data to S3
        m = re.search("//(.+)/", model_data_path)
        if m:
            bucket_name = m.group(1)
        print(f"bucket_name: {bucket_name}")

        object_name = "code.tar.gz"

        source_dir = f"s3://{bucket_name}/code/{object_name}"  # because on S3
        output_path = f"s3://{bucket_name}/"

        # Get parameters from the algorithm

        estimator = PyTorch(
            entry_point="script.py",
            source_dir=source_dir,
            role=aws_role,
            instance_count=1,
            instance_type=instance_type,
            framework_version="2.0.0",
            py_version="py310",
            hyperparameters=hyperparameters,
            output_path=output_path,
        )

        estimator.fit(wait=False)

        db_model_version.job_name = estimator.latest_training_job.job_name

        # setattr(db_model_version, key, value)
        session.add(db_model_version)
        session.commit()
        session.refresh(db_model_version)

        return db_model_version


@merlion_router.get("/model_versions/{model_version_id}/status")
async def get_train_status(model_version_id: int):
    sagemaker_client = boto3.client("sagemaker")

    with Session(engine) as session:
        db_model = session.get(ModelVersion, model_version_id)
        if not db_model:
            raise HTTPException(status_code=404, detail="ModelVersion not found")

    sm_job_name = db_model.job_name

    model_version_status = ModelVersionStatus.TrainingNotStarted
    if sm_job_name is not None:
        # Suppose 'your_training_job_name' is the name of your training job
        response = sagemaker_client.describe_training_job(TrainingJobName=sm_job_name)
        model_version_status = f"Training{response['TrainingJobStatus']}"

    if model_version_status == ModelVersionStatus.TrainingCompleted:
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=sm_job_name)
            endpoint_status = f"Endpoint{response['EndpointStatus']}"
        except:
            endpoint_status = None

    return JSONResponse(
        {"status": model_version_status if endpoint_status is None else endpoint_status}
    )


@merlion_router.get("/model_versions/{model_version_id}/deploy")
async def deploy_model_version(model_version_id: int):
    sagemaker_client = boto3.client("sagemaker")

    with Session(engine) as session:
        db_model = session.get(ModelVersion, model_version_id)
        if not db_model:
            raise HTTPException(status_code=404, detail="ModelVersion not found")

    sm_job_name = db_model.job_name

    model_data_path = os.getenv("MODEL_DATA_PATH", None)
    aws_role = os.getenv("AWS_ROLE", None)
    instance_type = os.getenv("INSTANCE_TYPE", None)

    # Specify the S3 location of your model.tar.gz file
    model_data = f"{model_data_path}{sm_job_name}/output/model.tar.gz"

    # Create a PyTorchModel object
    model = PyTorchModel(
        model_data=model_data,
        role=aws_role,
        framework_version="2.0.0",
        py_version="py310",
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=sm_job_name,
    )

    return JSONResponse({"status": ModelVersionStatus.CreatingEndpoint})


@merlion_router.get("/model_versions/{model_version_id}/undeploy")
async def undeploy_model_version(model_version_id: int):
    sagemaker_client = boto3.client("sagemaker")

    with Session(engine) as session:
        db_model = session.get(ModelVersion, model_version_id)
        if not db_model:
            raise HTTPException(status_code=404, detail="ModelVersion not found")

    sm_job_name = db_model.job_name

    job_status = "NotStarted"
    if sm_job_name is not None:
        # Suppose 'your_training_job_name' is the name of your training job
        response = sagemaker_client.describe_training_job(TrainingJobName=sm_job_name)
        job_status = response["TrainingJobStatus"]

    return JSONResponse({"status": job_status})


@merlion_router.post("/model_versions/{model_version_id}/predict")
def predict(model_version_id: int, data: dict):
    pass
