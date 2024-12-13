""" En este modulo se define la funcionalidad para el agente de licitaciones. """

import os
import asyncio
import logging
from pydantic import BaseModel
from typing import List, Dict

from langchain_core.output_parsers import JsonOutputParser, CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openai import RateLimitError
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma

from ..internal.path_utils import TEMP_DIR, CHROMA_DB, CHROMA_COLLECTION
from ..internal.prompt_utils import load_prompt
from .rag_utils import load_document_chroma

router = APIRouter()

logger = logging.getLogger(__name__)

llm = AzureChatOpenAI(
    azure_deployment='gpt-4o',
    api_version="2024-08-01-preview",
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.2  # Ajusta según tus necesidades

)

embeddings = AzureOpenAIEmbeddings(
    model="LLM-Codexca_text-embedding-3-large",
    dimensions=None,
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version="2023-05-15"
)


class RelevantDocumentRequest(BaseModel):
    """Modelo para la solicitud de documentos relevantes."""
    query: str
    num_proposals: int = 5


class ConceptNotesRequest(BaseModel):
    """Modelo para la solicitud de concept notes."""
    proposal_id: str
    tdr_summary: str
    information_sources: List[str]


class SaveConceptNotesRequest(BaseModel):
    """Modelo para la solicitud de concept notes."""
    proposal_id: str
    concept_notes: Dict


class TenderProposal(BaseModel):
    """Modelo para las propuestas de proyecto."""
    proposal_id: str
    tdr_summary: str = None
    title: str = None
    information_sources: List[str] = None
    concept_notes: Dict[str, str] = None
    index: List[str] = None
    key_ideas: str = None
    content: Dict[str, str] = None

    def get_proposal_id(self):
        return self.proposal_id

    def set_proposal_id(self, proposal_id: str):
        self.proposal_id = proposal_id

    def get_tdr_summary(self):
        return self.tdr_summary

    def set_tdr_summary(self, tdr_summary: str):
        self.tdr_summary = tdr_summary

    def get_title(self):
        return self.title

    def set_title(self, title: str):
        self.title = title

    def get_information_sources(self):
        return self.information_sources

    def set_information_sources(self, information_sources: List[str]):
        self.information_sources = information_sources

    def get_concept_notes(self):
        return self.concept_notes

    def set_concept_notes(self, concept_notes: str):
        self.concept_notes = concept_notes

    def get_index(self):
        return self.index

    def set_index(self, index: List[str]):
        self.index = index

    def get_key_ideas(self):
        return self.key_ideas

    def set_key_ideas(self, key_ideas: str):
        self.key_ideas = key_ideas

    def get_content(self):
        return self.content

    def set_content(self, content: Dict[str, str]):
        self.content = content


proposals: Dict[str, TenderProposal] = {}


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


@router.post("/upload-tdr/")
async def upload_tdr(file: UploadFile = File(...), num_proposals: int = 5):
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
    related_project = make_project_proposal(
        RelevantDocumentRequest(query=summary.content,
                                num_proposals=num_proposals),
        from_endpoint=False)

    os.remove(temp_file_path)

    return JSONResponse({"key points": tender_data, "complete summary": summary.content,
                         "related projects": related_project})

# TODO - Que se devuelvan los proyectos unicos.


@router.post('/related-projects/')
def make_project_proposal(request: RelevantDocumentRequest, from_endpoint: bool = False):
    """ Obtiene X propuestas de proyecto a partir de los términos de referencia 
        y la información extraída. """

    query = request.query
    num_proposals = request.num_proposals
    vector_store = Chroma(embedding_function=embeddings,
                          persist_directory=CHROMA_DB,
                          collection_name=CHROMA_COLLECTION)

    retriever = vector_store.as_retriever(
        search_type='mmr', search_kwargs={'k': num_proposals})

    related_docs = retriever.invoke(query)
    related_docs = [{str(doc.metadata['source']).split(sep='\\')[-1]: doc.page_content}
                    for doc in related_docs]
    if from_endpoint:
        return JSONResponse({"related_projects": related_docs, "query": query})
    return related_docs

# TODO - Rediseñar la forma en que se cargan los documentos de información.
# TODO - Modificar la respuesta para que se devuelva un JSON con los documentos y sus fuentes.


@router.post('/make-concept-notes/')
def make_concept_notes(request: ConceptNotesRequest):
    """ Genera las notas conceptuales de un proyecto. """

    proposal_id = request.proposal_id
    information_sources = request.information_sources
    tdr_summary = request.tdr_summary

    logger.info('Creating proposal with ID: %s', proposal_id)
    proposals[proposal_id] = TenderProposal(
        proposal_id=proposal_id)  # Crea una nueva propuesta
    proposal = proposals[proposal_id]
    proposal.set_information_sources(information_sources)
    proposal.set_tdr_summary(tdr_summary)
    logger.info('Generating concept notes for proposal: %s. Information sources: %s',
                proposal_id, str(information_sources))
    information_source_docs = [load_document_chroma(source)
                               for source in information_sources]

    json_template = load_prompt(prompt_name="make_concept_notes")
    json_parser = JsonOutputParser()
    concept_notes_prompt = PromptTemplate(
        template=(json_template),
        input_variables=['tdr_summary', 'information_sources'],
        partial_variables={
            "format_instructions": json_parser.get_format_instructions(),
        }
    )
    chain = concept_notes_prompt | llm | json_parser
    try:
        response = chain.invoke(
            {"tdr_summary": tdr_summary, "information_sources": information_source_docs})
    except RateLimitError as e:
        raise HTTPException(
            status_code=429,
            detail="Documento demasiado extenso para el modelo. Pruebe con un documento más corto.") from e
    return JSONResponse({"concept_notes": response})


@router.post('/save-concepts-notes/')
def save_concept_notes(request: SaveConceptNotesRequest):
    """ Guarda las notas conceptuales de un proyecto a partir de su ID. """

    proposal_id = request.proposal_id
    concept_notes = request.concept_notes
    proposals[proposal_id].set_concept_notes(concept_notes)
    return JSONResponse(status_code=200, content={"message": "Notas conceptuales guardadas con éxito."})


@router.get('/make-index/')
def make_index(proposal_id):
    """ Genera un índice de un proyecto. """
    proposal = proposals[proposal_id]
    concept_notes = proposal.get_concept_notes()
    tdr_summary = proposal.get_tdr_summary()

    json_template = load_prompt(prompt_name="make_index")
    json_parser = JsonOutputParser()

    index_prompt = PromptTemplate(
        template=(json_template),
        input_variables=['tdr_summary', 'concept_notes'],
        partial_variables={
            "format_instructions": json_parser.get_format_instructions(),
        }
    )

    chain = index_prompt | llm | json_parser
    response = chain.invoke(
        {"tdr_summary": tdr_summary, "concept_notes": concept_notes})
    return response


@router.post('/save-index/')
def save_index(proposal_id: str, index: List[str]):
    """ Guarda el índice de un proyecto a partir de su ID. """
    proposals[proposal_id].set_index(index)
    return JSONResponse(status_code=200, content={"message": "Índice guardado con éxito."})


@router.get('/proposal-trace/')
def get_proposal_trace(proposal_id: str):
    """ Obtiene el registro de una propuesta a partir de su ID."""
    
    raise HTTPException(status_code=503, detail="Este endpoint está en desarrollo. Por favor, inténtelo más tarde.")

    # try:
    #     proposal = proposals[proposal_id]
    # except KeyError:
    #     return JSONResponse(status_code=404, content={"message": "Código de propuesta no encontrado."})
    # return JSONResponse(proposal.model_dump(mode='json'))


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

# TODO - Ofrecer un outline con la estructura de la propuesta donde el usuario puede escoger, modificar y reordenar.
