""" Módulo con funciones para la gestión de prompts"""

import logging
from fastapi import HTTPException
import orjson as json
from .path_utils import SYSTEM_PROMPT_PATH

logger = logging.getLogger(__name__)


def load_prompt(file_path: str = SYSTEM_PROMPT_PATH, prompt_name: str = None):
    """ Lee un prompt o esquema específico del sistema desde un archivo JSON."""
    if prompt_name:
        try:
            with open(file_path, 'rb') as file:
                prompts = json.loads(file.read())
            logger.info("Prompt -> %s", prompt_name)
            return prompts.get(prompt_name, None)
        except FileNotFoundError:
            logger.info("El archivo %s no se encontró.", file_path)
            return None
        except json.JSONDecodeError:
            logger.info("Error al decodificar el archivo JSON %s.", file_path)
            return None
    else:
        raise ValueError("El nombre del prompt es requerido.")


def show_prompt(prompt_name: str = None):
    """ Muestra un prompt específico del sistema desde un archivo JSON. 
        Si no se especifica un nombre, muestra todos los prompts."""
    try:
        with open(SYSTEM_PROMPT_PATH, 'rb') as file:
            data = json.loads(file.read())
        logger.info("Prompt consultado -> %s", prompt_name)

        if prompt_name:
            logger.info("Prompt consultado -> %s", prompt_name)
            return data.get(prompt_name, None)
        logger.info("Prompts consultados -> %s", data.keys())
        return data
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"El archivo {SYSTEM_PROMPT_PATH} no se encontró.") from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Error al decodificar el archivo JSON {SYSTEM_PROMPT_PATH}.") from exc