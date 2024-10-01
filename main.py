import logging
import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter
from langchain_openai import ChatOpenAI

from .routers import chat

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargamos las variables de entorno
load_dotenv()

app = FastAPI()
app.include_router(chat.router)


@app.get("/")
def read_root():
    return {"Status": "API is running"}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
