"""Este módulo contiene la lógica para los endpoint de la funcionalidad de chat."""

import os
import logging

from typing import List
from datetime import datetime
from fastapi import HTTPException, APIRouter, Query
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

router = APIRouter()

# TODO Plantear un modelo más eficiente.
# global conversations

# conversations: Dict[str, Dict]
conversations = {}


class Message(BaseModel):
    """Modelo para los mensajes de chat."""
    role: str  # El rol del mensaje, puede ser "human" o "system"
    content: str = None
    timestamp: datetime = Field(datetime.now(), alias="timestamp")


class Chat(BaseModel):
    """Modelo para los mensajes de chat."""
    conversation_id: str
    messages: List[Message]


llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2  # Ajusta según tus necesidades
)


def get_response_langchain(prompt: str, stream=False):
    """Función básica que obtiene una respuesta desde Langchain."""
    if not prompt:
        prompt = "Dime un chiste sobre programación"

    system_prompt = (
        # "You are an assistant for question-answering tasks. "
        # "Use the following pieces of retrieved context to answer "
        # "the question. If you don't know the answer, say that you "
        # "don't know."
        # "\n\n"
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
        '''
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", f"{prompt}"),
        ]
    )

    logger.info("Obteniendo respuesta desde Langchain para el mensaje: %s",
                qa_prompt.format_messages())
    try:
        response = llm.invoke(qa_prompt.format_messages(), stream=stream)
        # for chunk in response:
        #     yield text_chunk
        return response.content
    except Exception as e:
        logger.error("Error al obtener la respuesta desde Langchain: %s", e)
        raise HTTPException(
            status_code=500, detail="Error al procesar la solicitud.") from e


def chat_langchain():
    pass

@router.post("/test_chat/")
async def test_chat(message: Message, stream: bool = Query(False)):
    """Endpoint para probar la funcionalidad de chat."""
    if stream:
        return StreamingResponse(get_response_langchain(prompt=message.content, stream=stream), media_type="text/plain")
    else:
        response = get_response_langchain(prompt=message.content)
        return Message(role="system", content=response, timestamp=datetime.now())


@router.post("/chat/")
async def chat_endpoint(chat_data: Chat):
    """Endpoint para enviar mensajes de chat."""
    conversation_id = chat_data.conversation_id
    messages = chat_data.messages

    if conversation_id not in conversations:
        conversations[conversation_id] = {
            "messages": [],
            "last_update": datetime.now()
        }
    else:
        conversations[conversation_id]["messages"].extend(messages)
        conversations[conversation_id]["last_update"] = datetime.now()

    # Obtener la respuesta de Langchain
    prompt = messages[-1].content if messages else ""
    response_content = get_response_langchain(prompt)

    # Añadir la respuesta al historial de la conversación
    conversations[conversation_id]["messages"].append(
        Message(role="system", content=response_content)
    )

    print("\nConversación completa: \n", conversations[conversation_id])
    return {"response": response_content}
