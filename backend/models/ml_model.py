from datetime import datetime
from typing import List, Optional

# One line of FastAPI imports here later 👈
from sqlmodel import Column, Field, JSON, Session, SQLModel, create_engine, select



class ModelBase(SQLModel):
    name: str = Field(index=True)
    description: Optional[str] = Field(default=None, index=True)
    datasource_id: int
    tag_names: List[str] = Field(sa_column=Column(JSON))

class Model(ModelBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

class ModelCreate(ModelBase):
    pass

class ModelRead(ModelBase):
    id: int

class ModelUpdate(SQLModel):
    name: Optional[str] = None
    description: Optional[str] = None
    datasource_id: Optional[int] = None
    tag_names: Optional[List[str]] = None


class ModelVersion(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    model_version_name: str
    model_id: int
    datasource_id: int
    start_datetime: datetime
    end_datetime: datetime
    train_test_split: float
