from datetime import datetime
from typing import Dict, List, Optional

from database import engine
# One line of FastAPI imports here later 👈
from sqlmodel import (
    Enum,
    JSON,
    Column,
    Field,
    Session,
    SQLModel,
    select,
    Relationship
)


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


class AlgorithmName(str, Enum):
    AutoEncoder = "AutoEncoder"
    LSTMED = "LSTMED"
    VAE = "VAE"


class ModelVersionStatus(str, Enum):
    TrainingNotStarted = "TrainingNotStarted"
    TrainingInProgress = "TrainingInProgress"
    TrainingCompleted = "TrainingCompleted"
    TrainingFailed = "TrainingFailed"
    CreatingEndpoint = "CreatingEndpoint"
    UpdatingEndpoint = "UpdatingEndpoint"
    RollingBackEndpoint = "RollingBackEndpoint"
    InServiceEndpoint = "InServiceEndpoint"
    DeletingEndpoint = "DeletingEndpoint"
    FailedEndpoint = "FailedEndpoint"


class ModelVersionBase(SQLModel):
    name: str
    job_name: Optional[str] = Field(default=None, index=True)
    datasource_id: int
    start_datetime: datetime = datetime(1970, 1, 1)
    end_datetime: datetime = datetime(2023, 7, 20)
    train_test_split: float = 0.8
    algorithm_name: AlgorithmName = AlgorithmName.LSTMED
    algorithm_parameters: Dict[str, str] = Field(default=dict(), sa_column=Column(JSON))

    model_id: int = Field(foreign_key="model.id")


class ModelVersion(ModelVersionBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)


class ModelVersionCreate(ModelVersionBase):
    pass


class ModelVersionRead(ModelVersionBase):
    id: int


class ModelVersionUpdate(SQLModel):
    name: Optional[str] = None
    datasource_id: Optional[int] = None
    start_datetime: Optional[datetime] = None
    end_datetime: Optional[datetime] = None
    train_test_split: Optional[float] = None
    algorithm_name: Optional[AlgorithmName] = None
    algorithm_parameters: Optional[Dict[str, str]] = None


def create_model_and_model_version():
    with Session(engine) as session:
        db_model = session.exec(select(Model)).first()
        if not db_model:
            model_1 = Model(
                name="model_1",
                description="model_1 description",
                datasource_id=1,
                tag_names=[
                    "CoolerTemp",
                    "BathTemp",
                    "CoolerSwitch",
                    "RefridgentTemp",
                    "CompressorCurrent",
                ],
            )
            session.add(model_1)
            session.commit()

        db_model = session.exec(select(Model)).first()

        db_model_version = session.exec(select(ModelVersion)).first()

        if not db_model_version:
            model_version_1 = ModelVersion(
                name="model_version_1",
                datasource_id=16,
                start_datetime=datetime(2022, 10, 14, 0, 1, 0),
                end_datetime=datetime(2023, 3, 22, 15, 48, 0),
                train_test_split=0.8,
                model_id=db_model.id,
            )

            session.add(model_version_1)
            session.commit()


class ModelSchedulerBase(SQLModel):
    model_version_id: int = Field(foreign_key="modelversion.id")
    start_time: datetime
    seconds_to_repeat: int
    datasource_id: int


class ModelScheduler(ModelSchedulerBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    model_version: Optional[ModelVersion] = Relationship()



class ModelSchedulerCreate(ModelSchedulerBase):
    pass


class ModelSchedulerRead(ModelSchedulerBase):
    id: int
