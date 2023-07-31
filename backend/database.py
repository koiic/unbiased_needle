import os

from dotenv import load_dotenv
from sqlmodel import SQLModel, create_engine

load_dotenv()


db_engine_url = os.getenv("DB_ENGINE_URL", "sqlite:///database.db")

print(db_engine_url, "db_engine_url")
engine = create_engine(db_engine_url, echo=True)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
