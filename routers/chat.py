"""Este módulo contiene la lógica para los endpoint de la funcionalidad de chat."""

import os
import logging

from typing import Dict, List
from datetime import datetime
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from ..internal.path_utils import CHROMA_DB, CHROMA_COLLECTION
from ..internal.prompt_utils import load_prompt, show_prompt

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
    retrieved_documents: List = None


conversations: Dict[str, BaseChatMessageHistory] = {}

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2  # Ajusta según tus necesidades
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

vector_store = Chroma(embedding_function=embeddings,
                      persist_directory=CHROMA_DB,
                      collection_name=CHROMA_COLLECTION)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Función para obtener el historial de mensajes de una conversación."""
    if session_id not in conversations:
        conversations[session_id] = InMemoryChatMessageHistory()
    return conversations[session_id]


@router.get("/prompt/")
def get_prompt(prompt_name: str = None):
    """Función para obtener un prompt específico. Si no se especifica un nombre, 
        muestra todos los prompts."""
    return show_prompt(prompt_name=prompt_name) if prompt_name else show_prompt()


@router.post("/chat/", deprecated=True)
async def chat_history_endpoint(request: ChatRequest):
    """Endpoint para interactuar con un modelo de lenguaje como un chatbot."""

    conversation_id = request.conversation_id
    user_input = request.prompt
    system_prompt = load_prompt(prompt_name='system_prompt_for_chat')
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


@router.get("/chat-trace/")
def get_chat_trace(session_id: str, prettier: bool = False):
    """Función para obtener el historial de mensajes de una conversación."""
    if session_id not in conversations:
        conversations[session_id] = InMemoryChatMessageHistory()
        logger.info(
            "No se encontró historial de mensajes para la conversación %s. Creando una nueva.",
            session_id)
    if prettier:
        formatted_history = []
        for message in conversations[session_id].messages:
            role = "Humano" if message.type == "human" else "Chany"
            formatted_message = {
                "role": role,
                "content": message.content,
            }
            formatted_history.append(formatted_message)
        return JSONResponse(content={"messages": formatted_history})
    return conversations[session_id]


@router.post("/chat-rag/")
async def chat_rag_endpoint(request: ChatRequest):
    """Pass"""
    conversation_id = request.conversation_id
    user_input = request.prompt
    config = {'configurable': {'session_id': conversation_id}}
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}, search_type="similarity")

    contextualize_q_system_prompt = load_prompt(
        prompt_name='contextualize_question_prompt')

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder('history'),
            ("human", "{input}"),

        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = load_prompt(prompt_name='qa_system_prompt')

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder('history'),
            ("human", "{input}"),

        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain)

    conversation_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer"
    )

    response = conversation_rag_chain.invoke(
        {'input': user_input}, config=config)
    return ChatResponse(
        response=response['answer'],
        conversation_id=conversation_id,
        retrieved_documents=response['context']
    )
