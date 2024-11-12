""" Módulo con funciones para la gestión de prompts"""

import logging
import orjson as json
from .path_utils import SYSTEM_PROMPT_PATH

logger = logging.getLogger(__name__)


def load_prompt(file_path: str = SYSTEM_PROMPT_PATH, prompt_name: str = None):
    """
    Lee un prompt específico del sistema desde un archivo JSON.

    :param file_path: Ruta del archivo JSON.
    :param prompt_name: Nombre del prompt a cargar.
    :return: El prompt solicitado o None si no se encuentra.
    """
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
