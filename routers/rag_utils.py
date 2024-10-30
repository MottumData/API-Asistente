import os
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse


router = APIRouter()


@router.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Endpoint para subir un archivo PDF."""
    if not file.filename.endswith(".pdf"):
        return JSONResponse(content={"error": "Únicamente se pueden adjuntar archivos PDF"}, status_code=400)

    directory = "rag_documents"

    file_path = os.path.join(directory, file.filename)
    if os.path.exists(file_path):
        return JSONResponse(content={"error": "El documento ya está en el RAG"}, status_code=400)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    return JSONResponse(content={"message": "Documento subido correctamente", "filename": file.filename})

