""" Módulo para la funcionalidad del RAG."""

import os
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from ..internal.path_utils import RAG_DIR, CHROMA_DB, CHROMA_COLLECTION


load_dotenv()

router = APIRouter()

logger = logging.getLogger(__name__)


def save_file(file: UploadFile, directory: str) -> str:
    """Guarda el archivo en el directorio especificado."""
    file_path = os.path.join(directory, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    logger.info("Archivo guardado en %s", file_path)
    return file_path


def delete_file(file_path: str) -> None:
    """Elimina el archivo especificado."""
    os.remove(file_path)
    logger.info("Archivo eliminado: %s", file_path)


def insert_document_chroma(file_path: str):
    """Función para insertar un documento en la base de datos de Chroma."""

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

    loader = PyPDFLoader(file_path=file_path)
    doc_data = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=300,
        chunk_overlap=50,
    )

    docs = splitter.split_documents(doc_data)

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DB,
        collection_name=CHROMA_COLLECTION)
    logger.info("Documento insertado en Chroma: %s", file_path)


def delete_document_chroma(nombre_archivo: str):
    """Elimina un documento de ChromaDB basado en su nombre de archivo."""

    # Inicializar las incrustaciones y la conexión a ChromaDB
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-smalls", api_key=os.getenv("OPENAI_API_KEY"))
    vectores = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DB,
        collection_name=CHROMA_COLLECTION
    )

    # Recuperar todos los documentos y sus metadatos
    documentos = vectores.get()
    metadatos = documentos['metadatas']
    ids = documentos['ids']

    # Buscar el ID del documento que coincide con el nombre de archivo
    id_a_eliminar = []
    for meta, id_doc in zip(metadatos, ids):
        if meta.get('source') == nombre_archivo:
            id_a_eliminar.append(id_doc)

    if id_a_eliminar:
        # Eliminar el documento de la colección
        vectores.delete(ids=id_a_eliminar)
        logger.info("Documento '%s' eliminado de ChromaDB. Con un total de '%s' identificadores ",
                    nombre_archivo, len(id_a_eliminar))
    else:
        logger.info("Documento '%s' no encontrado en ChromaDB.",
                    nombre_archivo)


def list_documents_chroma():
    """Función para listar los documentos en la base de datos de Chroma."""
    embedding = OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    client = chromadb.PersistentClient(CHROMA_DB)

    col = client.get_collection(
        name=CHROMA_COLLECTION, embedding_function=embedding)

    all_data = col.get(
        include=["documents", "metadatas"],
    )
    return all_data


@router.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Endpoint para subir un archivo PDF."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Únicamente se pueden adjuntar archivos PDF")

    file_path = os.path.join(RAG_DIR, file.filename)
    if os.path.exists(file_path):
        raise HTTPException(
            status_code=400, detail="El documento ya está en el RAG")

    try:
        save_file(file, RAG_DIR)
        insert_document_chroma(file_path)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al guardar el archivo: {str(e)}") from e

    return JSONResponse(content={"message": "Documento subido correctamente",
                                 "filename": file.filename})


@router.get("/list-documents/")
async def list_documents():
    """Endpoint para listar los documentos en el directorio que alimenta al RAG."""
    exclude_extensions = [".gitkeep"]

    documents = [f for f in os.listdir(RAG_DIR) if not any(
        f.endswith(ext) for ext in exclude_extensions)]

    return JSONResponse(content={"documents": documents})


@router.delete("/delete-document/")
async def delete_pdf(filename: str):
    """Endpoint para eliminar un documento del directorio que alimenta al RAG."""
    file_path = os.path.join(RAG_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404, detail="El documento no existe en el RAG")

    try:
        delete_file(file_path)
        delete_document_chroma(file_path)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al eliminar el archivo: {str(e)}") from e

    return JSONResponse(content={"message": "Documento eliminado correctamente",
                                 "filename": filename})


@router.get("/list-chroma-documents/")
async def list_chroma_documents():
    """Endpoint para listar los documentos en ChromaDB."""
    try:
        documents = list_documents_chroma()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al listar los documentos: {str(e)}") from e

    return JSONResponse(content={"documents": documents})
