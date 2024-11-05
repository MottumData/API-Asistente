"""Este módulo contiene la lógica para los endpoint de la funcionalidad de chat."""

import os
import logging

from typing import Dict
from datetime import datetime
from fastapi import HTTPException, APIRouter, Query
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter()

class Message(BaseModel):
    """Modelo para los mensajes de chat."""
    role: str  # El rol del mensaje, puede ser "human" o "system"
    content: str = None
    timestamp: datetime = Field(datetime.now(), alias="timestamp")


class ChatRequest(BaseModel):
    """Modelo para los mensajes de chat."""
    conversation_id: str
    prompt: str


class ChatResponse(BaseModel):
    """Modelo para los mensajes de respuesta del LLM"""
    conversation_id: str
    response: str


conversations: Dict[str, BaseChatMessageHistory] = {}

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2  # Ajusta según tus necesidades
)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Función para obtener el historial de mensajes de una conversación."""
    if session_id not in conversations:
        conversations[session_id] = InMemoryChatMessageHistory()
    return conversations[session_id]


def get_response_langchain(prompt: str, stream=False):
    """Función básica que obtiene una respuesta desde Langchain."""
    if not prompt:
        prompt = "Dime un como puedes ayudarme"

    system_prompt = (
        '''
        Eres un asistente de inteligencia artificial para gestión documental y 
        redacción en la empresa Codexca. Ayuda a los usuarios principiantes a buscar 
        información en documentos y redactar contenido nuevo de forma profesional, 
        con un tono amable y accesible.

        Búsqueda de información: Encuentra y resume información clave 
        en documentos según lo solicitado.
        Redacción: Genera borradores claros y profesionales.
        Soporte: Explica cada paso de forma sencilla y evita términos técnicos 
        innecesarios.
        Mantén un tono profesional y empático, promoviendo una 
        interacción fluida y accesible.
        También responderás a todo tipo de preguntas aunque no estén relacionadas con el tema.
        '''
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", f"{prompt}"),
            MessagesPlaceholder(variable_name='history'),
        ]
    )

    logger.info("Obteniendo respuesta desde Langchain para el mensaje: %s",
                qa_prompt.format_messages())
    try:
        chain = qa_prompt | llm
        conversation_chain = RunnableWithMessageHistory(
            chain, get_session_history, output_messages_key="response")
        response = conversation_chain.invoke({'input': prompt}, stream=stream)
        # for chunk in response:
        #     yield text_chunk
        return response.content
    except Exception as e:
        logger.error("Error al obtener la respuesta desde Langchain: %s", e)
        raise HTTPException(
            status_code=500, detail="Error al procesar la solicitud.") from e


@router.post("/test_chat/")
async def test_chat(message: Message, stream: bool = Query(False)):
    """Endpoint para probar la funcionalidad de chat."""
    if stream:
        return StreamingResponse(get_response_langchain(prompt=message.content, stream=stream), media_type="text/plain")
    else:
        response = get_response_langchain(prompt=message.content)
        return Message(role="ai", content=response, timestamp=datetime.now())


@router.post("/chat/")
async def chat_history_endpoint(request: ChatRequest):
    """Endpoint para interactuar con un modelo de lenguaje como un chatbot."""

    conversation_id = request.conversation_id
    user_input = request.prompt
    system_prompt = (
        '''
        Eres un asistente de inteligencia artificial para gestión documental y 
        redacción en la empresa Codexca. Ayuda a los usuarios principiantes a buscar 
        información en documentos y redactar contenido nuevo de forma profesional, 
        con un tono amable y accesible.

        Búsqueda de información: Encuentra y resume información clave 
        en documentos según lo solicitado.
        Redacción: Genera borradores claros y profesionales.
        Soporte: Explica cada paso de forma sencilla y evita términos técnicos 
        innecesarios.
        Mantén un tono profesional y empático, promoviendo una 
        interacción fluida y accesible.
        También responderás a todo tipo de preguntas aunque no estén relacionadas con el tema.
        '''
    )
    config = {'configurable': {'session_id': conversation_id}}

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder('history'),
        ]
    )

    chain = qa_prompt | llm | StrOutputParser()

    conversation_chain = RunnableWithMessageHistory(chain, get_session_history)

    response = conversation_chain.invoke({'input': user_input}, config=config)
    return ChatResponse(response=response, conversation_id=conversation_id)


@router.get("/chat_trace/")
def get_chat_trace(session_id: str):
    """Función para obtener el historial de mensajes de una conversación."""
    if session_id not in conversations:
        conversations[session_id] = InMemoryChatMessageHistory()
        logger.info(
            "No se encontró historial de mensajes para la conversación %s. Creando una nueva.", session_id)
    return conversations[session_id]
