import logging
import os
from typing import Awaitable, Callable

import firebase_admin
import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse, UJSONResponse
from sqlmodel import SQLModel, create_engine
from starlette.middleware.cors import CORSMiddleware

from models import ml_model
from routers.merlion import merlion_router
from routers.users import users_router

load_dotenv()

app = FastAPI(title="MAIO")

if not firebase_admin._apps:
    # cred = credentials.Certificate("serviceAccountKey.json") #get your service account keys from firebase
    firebase_admin.initialize_app()

router = APIRouter()
router.include_router(users_router, prefix="/users")
router.include_router(merlion_router, prefix="/ml")

app.include_router(router, prefix="/api/v1")

db_engine_url = os.getenv("DB_ENGINE_URL", "sqlite:///database.db")

engine = create_engine(db_engine_url, echo=True)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


if __name__ == "__main__":
    create_db_and_tables()
    uvicorn.run(app, host="0.0.0.0", port=8000)
