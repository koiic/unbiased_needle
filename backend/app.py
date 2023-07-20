import logging
import os
from typing import Awaitable, Callable

import firebase_admin
import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse, UJSONResponse
from starlette.middleware.cors import CORSMiddleware

from routers.merlion import merlion_router
from routers.users import users_router

from models.ml_model import create_model_and_model_version

from database import create_db_and_tables, engine

load_dotenv()

app = FastAPI(title="Unbiased_Needle", version="0.1.0")


@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    create_model_and_model_version()


if not firebase_admin._apps:
    # cred = credentials.Certificate("serviceAccountKey.json") #get your service account keys from firebase
    firebase_admin.initialize_app()

router = APIRouter()
router.include_router(users_router, prefix="/users")
router.include_router(merlion_router, prefix="/ml")

app.include_router(router, prefix="/api/v1")
