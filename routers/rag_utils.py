""" Módulo para la funcionalidad del RAG."""

import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from ..internal.path_utils import RAG_DIR

router = APIRouter()


def save_file(file: UploadFile, directory: str) -> str:
    """Guarda el archivo en el directorio especificado."""
    file_path = os.path.join(directory, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    return file_path


def delete_file(file_path: str) -> None:
    """Elimina el archivo especificado."""
    os.remove(file_path)


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
        # TODO - Implementar la lógica para guardar el embedding del documento
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
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al eliminar el archivo: {str(e)}") from e

    return JSONResponse(content={"message": "Documento eliminado correctamente",
                                 "filename": filename})
