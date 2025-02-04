""" Módulo para la funcionalidad del RAG."""

import os
import json
import logging
import time
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import (
    PyPDFLoader,
    JSONLoader,
    TextLoader,
    UnstructuredExcelLoader,
    CSVLoader
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from ..internal.path_utils import (
    RAG_DIR,
    CHROMA_DB,
    CHROMA_COLLECTION,
    STRUCTURE_DIRECTORY,
    STRUCTURE_FILE,
    ALLOWED_EXTENSIONS
)


load_dotenv()

router = APIRouter()

logger = logging.getLogger(__name__)


llm = AzureChatOpenAI(
    azure_deployment='gpt-4o',
    api_version="2024-08-01-preview",
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("AZURE_API_KEY"),
    temperature=0.2
)

# Para el caso en el que se consuma directamente desde OpenAI
# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     api_key=os.getenv("OPENAI_API_KEY") # Crear variable de entorno con la API Key de OpenAI
# )


embeddings = AzureOpenAIEmbeddings(
    model="LLM-Codexca_text-embedding-3-large",
    dimensions=None,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version="2023-05-15"
)

# Para el caso en el que se consuma directamente desde OpenAI
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-large",
#     api_key=os.getenv("OPENAI_API_KEY") # Crear variable de entorno con la API Key de OpenAI
# )

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
    # Generación de metadatos solo disponible para PDF por el momento.

    if not any(file_path.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise ValueError(
            "El archivo debe tener una de las siguientes extensiones: %s" % ALLOWED_EXTENSIONS)
    loader = None
    metadata = None
    if file_path.endswith(".json"):
        logger.info("Cargando archivo JSON: %s", file_path)
        loader = JSONLoader(file_path=file_path, jq_schema=".[] | .[]")
    elif file_path.endswith(".pdf"):
        logger.info("Cargando archivo PDF: %s", file_path)
        loader = PyPDFLoader(file_path=file_path)
        # metadata = generate_metadata(loader.load())
    elif file_path.endswith(".txt"):
        logger.info("Cargando archivo de texto: %s", file_path)
        loader = TextLoader(file_path=file_path, encoding="utf-8")
    elif file_path.endswith(".xls") or file_path.endswith(".xlsx"):
        logger.info("Cargando archivo Excel: %s", file_path)
        loader = UnstructuredExcelLoader(file_path=file_path)
    elif file_path.endswith(".csv"):
        logger.info("Cargando archivo CSV: %s", file_path)
        loader = CSVLoader(file_path=file_path)

    doc_data = loader.load()
    if metadata:
        for doc in doc_data:
            doc.metadata.update(metadata)
    if not doc_data:
        logger.error("No se cargó contenido del archivo: %s", file_path)
        return None

    splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Full width comma
            "\uff0e",  # Full width full stop
            "",
        ],
        chunk_size=400,
        chunk_overlap=50,
        add_start_index=True,
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
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_type="azure",

        api_base=os.getenv("AZURE_ENDPOINT"),
        model_name="LLM-Codexca_text-embedding-3-large",
        api_version="2023-05-15",

    )
    client = chromadb.PersistentClient(CHROMA_DB)

    col = client.get_collection(
        name=CHROMA_COLLECTION, embedding_function=embedding)

    all_data = col.get(
        include=["metadatas"],
    )

    sources = {}
    for meta in all_data['metadatas']:
        source = meta.get('source')
        num_pages = meta.get('page', 0)
        title = meta.get('title', 'Sin título')
        author = meta.get('author', 'No especificado')
        date = meta.get('date', 'No especificado')
        summary = meta.get('summary', 'No especificado')
        if source not in sources:
            sources[source] = {"chunks": 0, "num_pages": num_pages, "title": title,
                               "author": author, "date": date, "summary": summary}
        sources[source]["chunks"] += 1
        if num_pages > sources[source]["num_pages"]:
            sources[source]["num_pages"] = num_pages

    return sources


def load_document_chroma(source: str):
    """Función para cargar un documento de ChromaDB basado en su nombre de archivo."""
    embedding = OpenAIEmbeddingFunction(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_type="azure",

        api_base=os.getenv("AZURE_ENDPOINT"),
        model_name="LLM-Codexca_text-embedding-3-large",
        api_version="2023-05-15",

    )
    client = chromadb.PersistentClient(CHROMA_DB)

    col = client.get_collection(
        name=CHROMA_COLLECTION, embedding_function=embedding)

    all_data = col.get(
        include=["documents", "metadatas"],
    )
    # logger.info("Documentos en ChromaDB: %s", len(all_data['documents']))

    document_chunks = []
    for doc, metadata in zip(all_data['documents'], all_data['metadatas']):
        if source in metadata.get('source'):
            document_chunks.append(doc)

    if document_chunks:
        return document_chunks

    return None


def save_dir_structure(root_dir: str = STRUCTURE_DIRECTORY,
                       extension: str = "txt",
                       bulk_insert: bool = False):
    """Guarda la estructura de directorios en un archivo de texto o JSON, 
    y almacena los archivos permitidos en chroma_db."""
    if extension == 'txt':
        with open(STRUCTURE_FILE, 'w', encoding='utf-8') as f:
            for dirpath, _, filenames in os.walk(root_dir):
                for filename in filenames:
                    if filename.split('.')[-1] in ALLOWED_EXTENSIONS:
                        file_path = os.path.join(dirpath, filename)
                        f.write(f"{file_path}\n")
                        # Almacenar el archivo en chroma_db
                        if bulk_insert:
                            insert_document_chroma(file_path)
                            time.sleep(0.5)
                if not filenames:
                    f.write(f"{dirpath}:\n")
    elif extension == 'json':
        directory_structure = {}
        for dirpath, _, filenames in os.walk(root_dir):
            relative_path = os.path.relpath(dirpath, root_dir)
            allowed_files = [filename for filename in filenames if filename.split(
                '.')[-1] in ALLOWED_EXTENSIONS]
            directory_structure[relative_path] = allowed_files

            for filename in allowed_files:
                file_path = os.path.join(dirpath, filename)
                # Almacenar el archivo en chroma_db
                if bulk_insert:
                    insert_document_chroma(file_path)

        with open(STRUCTURE_FILE, 'w', encoding='utf-8') as f:
            json.dump(directory_structure, f, ensure_ascii=False, indent=4)

    logger.info("Estructura de directorios guardada en %s", STRUCTURE_FILE)
    insert_document_chroma(STRUCTURE_FILE)


def generate_metadata(doc):
    """Genera metadatos para un documento."""
    # V1
    # llm = ChatOpenAI(model="gpt-4o-mini",
    #                  api_key=os.getenv("OPENAI_API_KEY"))
    # prompt = PromptTemplate.from_template(
    #     "Extrae los metadatos optimizado para vector search del siguiente documento sin decir
    # nada más:\n\n{document}\n\nMetadatos:")
    # chain = prompt | llm
    # metadata = chain.invoke({
    #     "document": doc
    # })
    # V2 - Genera para cada documento
    schema = {
        "properties": {
            "title": {"type": "string"},
            "author": {"type": "string"},
            "date": {"type": "string"},
            "summary": {"type": "string", "description": "Resumen breve del documento"},
            "tags": {"type": "string"}
        },
        "required": ["title", "author", "date", "summary", "tags"]
    }
    parser = JsonOutputParser()
    prompt = PromptTemplate(
        template=(
            "Extract the information as specified in \\{schema}.\n"
            "{format_instructions}\n"
            "{context}\n"
        ),
        input_variables=['context'],
        partial_variables={
            "format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    metadata = chain.invoke({"context": doc, "schema": schema})
    logger.info("Metadatos generados: %s", metadata)
    return metadata


@ router.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    """Endpoint para subir un archivo  PDF, JSON, TXT, CSV o XLS."""
    if not any(file.filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400, detail="Sólo se pueden adjuntar archivos: PDF, JSON, TXT, CSV o XLS")

    file_path = os.path.join(RAG_DIR, file.filename)
    if os.path.exists(file_path):
        raise HTTPException(
            status_code=400, detail="El documento ya está en el RAG")

    try:
        save_file(file, RAG_DIR)  # Guardar el archivo en el directorio
    except Exception as e:
        delete_file(file_path)
        logger.info(
            "Eliminado archivo al no poder guardarlo en Chroma: %s", file)
        raise HTTPException(
            status_code=500, detail=f"Error al guardar el archivo: {str(e)}") from e
    try:
        insert_document_chroma(file_path)  # Insertar el documento en ChromaDB
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al insertar el documento en ChromaDB: {str(e)}") from e

    return JSONResponse(content={"message": "Documento subido correctamente",
                                 "filename": file.filename})


@ router.get("/list-documents/")
async def list_documents():
    """Endpoint para listar los documentos en el directorio que alimenta al RAG."""
    exclude_extensions = [".gitkeep"]

    documents = [f for f in os.listdir(RAG_DIR) if not any(
        f.endswith(ext) for ext in exclude_extensions)]

    return JSONResponse(content={"documents": documents})


@ router.delete("/delete-document/")
async def delete_pdf(filename: str):
    """Endpoint para eliminar un documento del directorio que alimenta al RAG.
    También elimina el documento de ChromaDB."""
    file_path = os.path.join(RAG_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404, detail="El documento no existe en el RAG")

    try:
        delete_file(file_path)  # Eliminar el archivo del directorio
        delete_document_chroma(file_path)  # Eliminar el documento de ChromaDB
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al eliminar el archivo: {str(e)}") from e

    return JSONResponse(content={"message": "Documento eliminado correctamente",
                                 "filename": filename})


@ router.get("/list-chroma-documents/")
async def list_chroma_documents():
    """Endpoint para listar los documentos en ChromaDB."""
    try:
        documents = list_documents_chroma()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al listar los documentos: {str(e)}") from e

    return JSONResponse(content={"documents": documents})


@router.get("/load-chroma-document/")
async def load_chroma_document(source: str):
    """Endpoint para cargar un documento de ChromaDB."""
    try:
        document = load_document_chroma(source)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al cargar el documento: {str(e)}") from e

    return JSONResponse(content={"document": document})


@ router.get("/save-structure/")
async def save_directory_structure(extension: str = "txt", bulk_insert: bool = False):
    """Endpoint para obtener la estructura de directorios y archivos para el RAG en TXT o JSON. """
    if extension not in ["json", "txt", ".json", ".txt"]:
        raise HTTPException(
            status_code=400, detail="La extensión debe ser txt o json")
    if not os.path.exists(STRUCTURE_DIRECTORY):
        raise HTTPException(
            status_code=404, detail="No se encontró la estructura de directorios. Contacta al administrador.")

    save_dir_structure(extension=extension, bulk_insert=bulk_insert)
    with open(STRUCTURE_FILE, 'r', encoding='utf-8') as f:
        structure = f.read()
    return JSONResponse(content={"structure": structure})
