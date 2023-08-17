import inspect
import json
import logging
import os
import re
import tempfile
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta
from enum import Enum
from typing import List

import boto3
import sagemaker
from database import engine
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sagemaker.pytorch import PyTorch, PyTorchModel
from sqlmodel import Session, select

from .ml_model import (
    Model,
    ModelCreate,
    ModelRead,
    ModelScheduler,
    ModelSchedulerCreate,
    ModelSchedulerRead,
    ModelUpdate,
    ModelVersion,
    ModelVersionCreate,
    ModelVersionRead,
    ModelVersionStatus,
    ModelVersionUpdate,
)

ml_router = APIRouter()


@ml_router.get("/default_parameters/{algorithm_name}")
async def default_parameters(algorithm_name: str):
    assert algorithm_name in [
        "VAE",
        "LSTMED",
        "AutoEncoder",
    ], "Algorithm name not found"

    if algorithm_name == "VAE":
        json_encoded_data = {
            "encoder_hidden_sizes": {"type": "Array", "default": [25, 10, 5]},
            "decoder_hidden_sizes": {"type": "Array", "default": [5, 10, 25]},
            "latent_size": {"type": "Number", "default": 5},  # was <class 'int'>
            "sequence_len": {"type": "Number", "default": 1},  # was <class 'int'>
            "kld_weight": {"type": "Number", "default": 1},  # was <class 'float'>
            "dropout_rate": {"type": "Number", "default": 0},  # was <class 'float'>
            "num_eval_samples": {"type": "Number", "default": 10},  # was <class 'int'>
            "lr": {"type": "Number", "default": 0.001},  # was <class 'float'>
            "batch_size": {"type": "Number", "default": 1024},  # was <class 'int'>
            "num_epochs": {"type": "Number", "default": 10},  # was <class 'int'>
            "max_score": {"type": "Number", "default": 1000},  # was <class 'int'>
            "enable_calibrator": {
                "type": "Boolean",
                "default": True,
            },  # was <class 'bool'>
            "enable_threshold": {
                "type": "Boolean",
                "default": True,
            },  # was <class 'bool'>
        }
    elif algorithm_name == "LSTMED":
        json_encoded_data = {
            "hidden_size": {"type": "Number", "default": 5},  # was <class 'int'>
            "sequence_len": {"type": "Number", "default": 20},  # was <class 'int'>
            "n_layers": {"type": "Array", "default": [1, 1]},  # was <class 'int'>
            "dropout": {"type": "Array", "default": [0, 0]},  # was <class 'int'>
            "lr": {"type": "Number", "default": 0.001},  # was <class 'float'>
            "batch_size": {"type": "Number", "default": 256},  # was <class 'int'>
            "num_epochs": {"type": "Number", "default": 10},  # was <class 'int'>
            "max_score": {"type": "Number", "default": 1000},  # was <class 'int'>
            "enable_calibrator": {
                "type": "Boolean",
                "default": True,
            },  # was <class 'bool'>
            "enable_threshold": {
                "type": "Boolean",
                "default": True,
            },  # was <class 'bool'>
        }
    elif algorithm_name == "AutoEncoder":
        json_encoded_data = {
            "hidden_size": {"type": "Number", "default": 5},  # was <class 'int'>
            "layer_sizes": {"type": "Array", "default": [25, 10, 5]},
            "sequence_len": {"type": "Number", "default": 1},  # was <class 'int'>
            "lr": {"type": "Number", "default": 0.001},  # was <class 'float'>
            "batch_size": {"type": "Number", "default": 512},  # was <class 'int'>
            "num_epochs": {"type": "Number", "default": 50},  # was <class 'int'>
            "max_score": {"type": "Number", "default": 1000},  # was <class 'int'>
            "enable_calibrator": {
                "type": "Boolean",
                "default": True,
            },  # was <class 'bool'>
            "enable_threshold": {
                "type": "Boolean",
                "default": True,
            },  # was <class 'bool'>
        }
    else:
        raise Exception("Algorithm not found")

    return JSONResponse(content=json_encoded_data)


@ml_router.post("/models", response_model=ModelRead)
def create_model(model: ModelCreate):
    with Session(engine) as session:
        db_model = Model.from_orm(model)
        session.add(db_model)
        session.commit()
        session.refresh(db_model)
        return db_model


@ml_router.get("/models", response_model=List[ModelRead])
def read_models(offset: int = 0, limit: int = Query(default=100, lte=100)):
    with Session(engine) as session:
        models = session.exec(select(Model).offset(offset).limit(limit)).all()
        return models


@ml_router.get("/models/{model_id}", response_model=ModelRead)
def read_model(model_id: int):
    with Session(engine) as session:
        model = session.get(Model, model_id)
        if not model:
            raise HTTPException(status_code=500, detail="Model not found")
        return model


@ml_router.patch("/models/{model_id}", response_model=ModelRead)
def update_model(model_id: int, model: ModelUpdate):
    with Session(engine) as session:
        db_model = session.get(Model, model_id)
        if not db_model:
            raise HTTPException(status_code=500, detail="Model not found")
        model_data = model.dict(exclude_unset=True)
        for key, value in model_data.items():
            setattr(db_model, key, value)
        session.add(db_model)
        session.commit()
        session.refresh(db_model)
        return db_model


@ml_router.post("/model_versions", response_model=ModelVersionRead)
def create_model_versions(model_version: ModelVersionCreate):
    with Session(engine) as session:
        db_model_version = ModelVersion.from_orm(model_version)
        session.add(db_model_version)
        session.commit()
        session.refresh(db_model_version)
        return db_model_version


@ml_router.get("/model_versions", response_model=List[ModelVersionRead])
def read_model_versions(offset: int = 0, limit: int = Query(default=100, lte=100)):
    with Session(engine) as session:
        model_versions = session.exec(
            select(ModelVersion).offset(offset).limit(limit)
        ).all()
        return model_versions


@ml_router.get("/model_versions/{model_version_id}", response_model=ModelVersionRead)
def read_model_version(model_version_id: int):
    with Session(engine) as session:
        model_version = session.get(ModelVersion, model_version_id)
        if not model_version:
            raise HTTPException(status_code=500, detail="ModelVersion not found")
        return model_version


@ml_router.patch("/model_versions/{model_version_id}", response_model=ModelVersionRead)
def update_model_version(model_version_id: int, model_version: ModelVersionUpdate):
    with Session(engine) as session:
        db_model_version = session.get(ModelVersion, model_version_id)
        if not db_model_version:
            raise HTTPException(status_code=500, detail="ModelVersion not found")
        model_version_data = model_version.dict(exclude_unset=True)
        for key, value in model_version_data.items():
            print(f"key: {key}, value: {value}")
            setattr(db_model_version, key, value)
        session.add(db_model_version)
        session.commit()
        session.refresh(db_model_version)
        return db_model_version


@ml_router.post(
    "/model_versions/{model_version_id}/train", response_model=ModelVersionRead
)
async def train_model_version(model_version_id: int):
    with Session(engine) as session:
        db_model_version = session.get(ModelVersion, model_version_id)
        if not db_model_version:
            raise HTTPException(status_code=500, detail="ModelVersion not found")

        if db_model_version.job_name is not None:
            raise HTTPException(status_code=500, detail="ModelVersion already trained")

        db_model = session.get(Model, db_model_version.model_id)
        if not db_model:
            raise HTTPException(status_code=500, detail="Model not found")

        keys = db_model_version.algorithm_parameters.keys()

        hyperparameters = dict()
        hyperparameters["algorithm_name"] = db_model_version.algorithm_name
        if "maio_instance_str" not in keys:
            raise HTTPException(status_code=500, detail="maio_instance_str not found")
        hyperparameters["maio_instance_str"] = db_model_version.algorithm_parameters[
            "maio_instance_str"
        ]
        if "maio_token" not in keys:
            raise HTTPException(status_code=500, detail="maio_token not found")
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
        try:
            estimator.fit(wait=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {e}")

        db_model_version.job_name = estimator.latest_training_job.job_name

        # setattr(db_model_version, key, value)
        session.add(db_model_version)
        session.commit()
        session.refresh(db_model_version)

        return db_model_version


async def retrieve_model_version_status(model_version_id: int):
    """get model_version status from sagemaker directly to avoid having
    to keep the db in sync with sagemaker

    Args:
        model_version_id (int): the id of the model version

    Raises:
        HTTPException: 500 if the model version is not found

    Returns:
        ModelVersionStatus: the status of the model version
    """
    instance_type = os.getenv("INSTANCE_TYPE", None)

    if instance_type == "local":
        sagemaker_client = sagemaker.local.LocalSagemakerClient()
    else:
        sagemaker_client = boto3.client("sagemaker")

    with Session(engine) as session:
        db_model = session.get(ModelVersion, model_version_id)
        if not db_model:
            raise HTTPException(status_code=500, detail="ModelVersion not found")

    sm_job_name = db_model.job_name

    model_version_status = ModelVersionStatus.TrainingNotStarted
    if sm_job_name is not None:
        if instance_type == "local":
            # This is a hack to get the status of the training job when running locally
            try:
                response = sagemaker_client.describe_training_job(
                    TrainingJobName=sm_job_name
                )
            except Exception as e:
                print(e)

            model_version_status = ModelVersionStatus.TrainingCompleted
        else:
            response = sagemaker_client.describe_training_job(
                TrainingJobName=sm_job_name
            )
            model_version_status = f"Training{response['TrainingJobStatus']}"

    endpoint_status = None
    if model_version_status == ModelVersionStatus.TrainingCompleted:
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=sm_job_name)
            endpoint_status = f"{response['EndpointStatus']}Endpoint"

        except:
            endpoint_status = None

    return model_version_status if endpoint_status is None else endpoint_status


@ml_router.get("/model_versions/{model_version_id}/status")
async def get_model_version_status(model_version_id: int):
    """get model_version status from sagemaker directly to avoid having
    to keep the db in sync with sagemaker

    Args:
        model_version_id (int): the id of the model version

    Raises:
        HTTPException: 500 if the model version is not found

    Returns:
        ModelVersionStatus: the status of the model version
    """
    model_version_status = await retrieve_model_version_status(model_version_id)

    return JSONResponse({"status": model_version_status})


@ml_router.get("/model_versions/{model_version_id}/deploy")
async def deploy_model_version(model_version_id: int):
    # Getting status of the model version
    model_version_status = await get_model_version_status(model_version_id)
    data = json.loads(model_version_status.body)

    # If the model version is not trained yet, we cannot deploy it
    if data["status"] != ModelVersionStatus.TrainingCompleted:
        raise HTTPException(
            status_code=500,
            detail="ModelVersion not trained yet. Please train the model first.",
        )

    # Getting the sagemaker client
    sagemaker_client = boto3.client("sagemaker")

    # Getting the model version of given id
    with Session(engine) as session:
        db_model = session.get(ModelVersion, model_version_id)
        if not db_model:
            raise HTTPException(status_code=500, detail="ModelVersion not found")

    sm_job_name = db_model.job_name

    model_data_path = os.getenv("MODEL_DATA_PATH", None)
    aws_role = os.getenv("AWS_ROLE", None)
    instance_type = os.getenv("INSTANCE_TYPE", None)

    # Specify the S3 location of your model.tar.gz file
    if instance_type == "local":
        model_data = f"{model_data_path}{sm_job_name}/model.tar.gz"
    else:
        model_data = f"{model_data_path}{sm_job_name}/output/model.tar.gz"

    # Create a PyTorchModel object
    model = PyTorchModel(
        model_data=model_data,
        role=aws_role,
        entry_point="script.py",
        source_dir="s3://maio-sagemaker/code_deploy/code.tar.gz",
        code_location="s3://maio-sagemaker/code_location/",
        framework_version="2.0.0",
        py_version="py310",
    )
    # Deploy the model to Amazon SageMaker
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=sm_job_name,
    )

    return JSONResponse({"status": ModelVersionStatus.CreatingEndpoint})


@ml_router.get("/model_versions/{model_version_id}/undeploy")
async def undeploy_model_version(model_version_id: int):
    # Getting status of the model version
    model_version_status = await get_model_version_status(model_version_id)
    data = json.loads(model_version_status.body)

    # If the model version is not trained yet, we cannot deploy it
    if data["status"] != ModelVersionStatus.InServiceEndpoint:
        raise HTTPException(
            status_code=500,
            detail="ModelVersion not deployed yet. Please deploy the model first.",
        )

    # Getting the sagemaker client
    sagemaker_client = boto3.client("sagemaker")

    # Getting the model version of given id
    with Session(engine) as session:
        db_model = session.get(ModelVersion, model_version_id)
        if not db_model:
            raise HTTPException(status_code=500, detail="ModelVersion not found")

    # Getting the job name of the model version (the endpoint name will be the same)
    sm_job_name = db_model.job_name

    # Deleting the endpoint
    sagemaker_client.delete_endpoint(EndpointName=sm_job_name)
    sagemaker_client.delete_endpoint_config(EndpointConfigName=sm_job_name)

    return JSONResponse({"status": ModelVersionStatus.DeletingEndpoint})


@ml_router.post("/model_versions/{model_version_id}/predict")
async def model_versions_predict(model_version_id: int, dt: datetime):
    model_version_status = await get_model_version_status(model_version_id)

    data = json.loads(model_version_status.body)

    if data["status"] != ModelVersionStatus.InServiceEndpoint:
        raise HTTPException(
            status_code=500,
            detail="ModelVersion not deployed yet. Please deploy the model first.",
        )

    # Getting the model version of given id
    with Session(engine) as session:
        model_version = session.get(ModelVersion, model_version_id)
        if not model_version:
            raise HTTPException(status_code=500, detail="ModelVersion not found")

    if dt < model_version.end_datetime:
        raise HTTPException(
            status_code=500,
            detail="dt is before end_datetime. Please provide a dt not in the training set.",
        )

    sagemaker_client = boto3.client("sagemaker-runtime")

    payload = {
        "datasource_id": model_version.datasource_id,
        "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "batch_size": 256,
    }

    json_payload = json.dumps(payload)

    # Invoke the SM endpoint
    response = sagemaker_client.invoke_endpoint(
        EndpointName=model_version.job_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json_payload,
    )

    # # Transform the response to a string
    # data = response["Body"].read().decode("utf-8")

    # anom_score = pd.read_json(json.loads(data), orient="split")

    return JSONResponse({"status": True})


@ml_router.post(
    "/model_versions/{model_version_id}/schedule", response_model=ModelSchedulerRead
)
async def schedule_model_version(
    model_version_id: int, model_scheduler: ModelSchedulerCreate
):
    with Session(engine) as session:
        model_version = session.get(ModelVersion, model_version_id)
        if not model_version:
            raise HTTPException(status_code=500, detail="ModelVersion not found")

        # check if model version is deployed
        model_version_status = await retrieve_model_version_status(model_version_id)

        if model_version_status != ModelVersionStatus.InServiceEndpoint:
            raise HTTPException(
                status_code=500,
                detail="ModelVersion not deployed yet. Please deploy the model first.",
            )

        model_scheduler.model_version_id = model_version_id
        db_model_scheduler = ModelScheduler.from_orm(model_scheduler)
        session.add(db_model_scheduler)
        session.commit()
        session.refresh(db_model_scheduler)

        lambda_function_arn = os.getenv(
            "LAMBDA_FUNCTION_ARN",
            "arn:aws:lambda:eu-west-1:146915812621:function:test_func_v2",
        )

        event_client = boto3.client("events")
        # if seconds_to_repeat = set to current time

        payload = {
            "datasource_id": db_model_scheduler.datasource_id,
            "token": model_version.algorithm_parameters["maio_token"],
            "endpoint": model_version.job_name,
            "start_time": datetime.strftime(
                db_model_scheduler.start_time, "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
        }

        if db_model_scheduler.seconds_to_repeat == 0:
            # TODO: call the lambda function immediately
            return db_model_scheduler

        rate = f"rate({db_model_scheduler.seconds_to_repeat // 60} minutes)"

        response = event_client.put_rule(
            Name=f"{db_model_scheduler.model_version.job_name}_{db_model_scheduler.id}",
            ScheduleExpression=rate,  # Set the desired interval
            State="ENABLED",
        )

        # Create the target for the rule
        target = {
            "Id": f"{db_model_scheduler.model_version.job_name}_{db_model_scheduler.id}",
            "Arn": lambda_function_arn,
            "Input": json.dumps(payload),  # Set the desired payload for each invocation
        }

        # Add the target to the rule
        event_client.put_targets(
            Rule=f"{db_model_scheduler.model_version.job_name}_{db_model_scheduler.id}",
            Targets=[target],
        )

        # Add permission for CloudWatch Events to invoke the Lambda function
        lambda_client = boto3.client("lambda")
        lambda_client.add_permission(
            FunctionName=lambda_function_arn.split(":")[-1],
            StatementId=f"{db_model_scheduler.model_version.job_name}_{db_model_scheduler.id}",
            Action="lambda:InvokeFunction",
            Principal="events.amazonaws.com",
            SourceArn=response["RuleArn"],
        )

        print("CloudWatch Event rule created successfully.")

    return db_model_scheduler


@ml_router.delete("/model_schedulers/{model_scheduler_id}")
async def delete_schedule(model_scheduler_id: int):
    with Session(engine) as session:
        model_scheduler = session.get(ModelScheduler, model_scheduler_id)
        if model_scheduler is None:
            raise HTTPException(status_code=500, detail="ModelScheduler not found")

        event_client = boto3.client("events")
        event_client.remove_targets(
            Rule=f"{model_scheduler.model_version.job_name}_{model_scheduler.id}",
            Ids=[f"{model_scheduler.model_version.job_name}_{model_scheduler.id}"],
        )

        event_client.delete_rule(
            Name=f"{model_scheduler.model_version.job_name}_{model_scheduler.id}"
        )

        session.delete(model_scheduler)
        session.commit()

    return JSONResponse({"status": True})
