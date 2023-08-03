from api.api_v1.api import router as api_router
from api.api_v1.endpoints.ml_model import create_model_and_model_version
from database import create_db_and_tables, engine
from dotenv import load_dotenv
from fastapi import FastAPI
from mangum import Mangum

load_dotenv()

app = FastAPI()


@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    create_model_and_model_version()


@app.get("/ping")
async def root():
    return {"message": "pong"}


app.include_router(api_router, prefix="/api/v1")
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
