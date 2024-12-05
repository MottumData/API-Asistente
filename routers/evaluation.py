""" Este módulo contiene la lógica para evaluar el modelo con el gold standard. """
import uuid
import logging
from datetime import datetime
import requests
import orjson as json
from fastapi import APIRouter,  HTTPException
from fastapi.responses import FileResponse


from ..internal.path_utils import GOLD_STANDARD_PATH

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/gold_standard_evaluation")
def gold_standard_evaluation(url="localhost:8000"):
    """ Evalúa el modelo con el gold standard y lo guarda en un archivo JSON con la fecha actual."""
    try:
        with open(GOLD_STANDARD_PATH, "rb") as json_file:
            data = [json.loads(line) for line in json_file]
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail="El archivo gold standard no se encontró.") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="Error al decodificar el archivo JSON.") from exc

    for row in data:
        playload = {
            "conversation_id": str(uuid.uuid4()),
            "prompt": row["question"],
        }
        response = requests.post(f"http://{url}/chat-rag/", json=playload)
        if response.status_code == 200:
            response_data = response.json()
            logger.info("Question: %s", row['question'])
            logger.info("Response: %s", response_data['response'])
            logger.info("-------------------------------------------------")
            row['respuesta_modelo'] = response_data['response']
            # Save the updated data to a new file with the current date in the filename

    new_filename = f"gold_standard_{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(new_filename, "wb") as outfile:
        for entry in data:
            outfile.write(json.dumps(entry) + b'\n')

    return FileResponse(new_filename, media_type='application/json', filename=new_filename)
