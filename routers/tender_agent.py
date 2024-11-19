""" En este modulo se define la funcionalidad para el agente de licitaciones. """

import os
import asyncio
import logging
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_community.document_loaders import PyPDFLoader

from ..internal.path_utils import TEMP_DIR
from ..internal.prompt_utils import load_prompt

router = APIRouter()

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# TODO - Diseñar el resto de elementos


def get_tender_data(doc):
    """ Obtiene los datos especificados en el esquema de las licitaciones. """
    schema = load_prompt(prompt_name="tender_schema")
    json_template = load_prompt(prompt_name="extract_tender_data")
    parser = JsonOutputParser()

    prompt = PromptTemplate(
        template=(json_template),
        input_variables=['schema', 'format_instructions', 'document'],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
        }
    )
    chain = prompt | llm | parser
    tdr_data = chain.invoke({"document": doc, "schema": schema})
    return tdr_data


def make_summary(doc, tender_data):
    """ Genera un resumen  a partir de los términos de referencia y la información extraída. """
    json_template = load_prompt(prompt_name="make_tender_summary")
    summary_prompt = PromptTemplate(
        template=(json_template),
        input_variables=['document', 'tender_data'],
    )
    chain = summary_prompt | llm
    summary = chain.invoke({"document": doc, "tender_data": tender_data})
    return summary

# TODO - Modificar esta funcion


@router.post("/upload-tdr/")
async def upload_tdr(file: UploadFile = File(...)):
    """ Sube un documento y lo guarda temporalmente en el directorio temp. """

    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    temp_file_path = os.path.join(TEMP_DIR, file.filename)

    with open(temp_file_path, "wb") as temp_file:
        contents = await file.read()
        temp_file.write(contents)

    loader = PyPDFLoader(temp_file_path)
    document = loader.load()
    tender_data = get_tender_data(document)
    summary = make_summary(document, tender_data)

    # Elimina el archivo temporal después de procesarlo
    os.remove(temp_file_path)

    return JSONResponse({"key points": tender_data, "complete summary": summary.content})


@router.post("/upload-tdr-streaming/")
async def upload_tdr_streaming(file: UploadFile = File(...)):
    """ Sube un documento y lo guarda temporalmente en el directorio temp. """

    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    temp_file_path = os.path.join(TEMP_DIR, file.filename)

    with open(temp_file_path, "wb") as temp_file:
        contents = await file.read()
        temp_file.write(contents)

    loader = PyPDFLoader(temp_file_path)
    document = loader.load()

    async def generate():
        tender_data = get_tender_data(document)
        yield JSONResponse({"key points": tender_data}).body.decode()
        await asyncio.sleep(1)

        summary = make_summary(document, tender_data)
        yield JSONResponse({"complete summary": summary.content}).body.decode()
        await asyncio.sleep(1)

        # Elimina el archivo temporal después de procesarlo
        os.remove(temp_file_path)

    return StreamingResponse(generate(), media_type="application/json")

# TODO - Devolver referencias de los documentos mas alienados
# TODO - Seleccionar las referencias pasadas.
# TODO - Ofrecer una descripción de 2-3 párrafos sobre el objetivo del proyecto y aparte ofrecer highlights del enfoque o metodología que debe escoger.
# TODO - Ofrecer un outline con la estructura de la propuesta donde el usuario puede escoger, modificar y reordenar.
