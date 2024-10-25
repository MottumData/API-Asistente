import logging
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import chat

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargamos las variables de entorno
load_dotenv()

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:80",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)


@app.get("/")
def read_root():
    return {"Status": "API is running"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
