"""Este script configura y ejecuta una aplicación FastAPI con carga de variables de entorno. 
   Punto de entrada de la aplicación."""

import logging
from datetime import datetime
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import chat

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carga de las variables de entorno
load_dotenv()

app = FastAPI()

origins = [
    "http://localhost:80",
    "http://localhost:8000",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)


@app.get("/ping")
def ping():
    """
    Endpoint para verificar el estado de la API.
    Devuelve el estado y el tiempo actual.
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"Status": "API is running", "Time": current_time}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
