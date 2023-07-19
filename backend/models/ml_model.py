from datetime import datetime
from typing import Optional

# One line of FastAPI imports here later 👈
from sqlmodel import Field, Session, SQLModel, create_engine, select


class Model(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: Optional[str] = None
    datasource_id: int


class ModelVersion(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    model_version_name: str
    model_id: int
    datasource_id: int
    start_datetime: datetime
    end_datetime: datetime
    train_test_split: float
