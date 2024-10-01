import logging

from fastapi import HTTPException, APIRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

router = APIRouter()


class Message(BaseModel):
    content: str = None


llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0  # Ajusta según tus necesidades
)


def get_response_langchain(prompt: str) -> str:
    if not prompt:
        prompt = "Dime un chiste sobre programación"
    try:
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know."
            "\n\n"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", f"{prompt}"),
            ]
        )

        # Ahora llamamos al modelo de chat correctamente
        logger.info(f"Obteniendo respuesta desde Langchain: {qa_prompt.format_messages()}")
        response = llm.invoke(qa_prompt.format_messages())  # El mensaje debe ser una lista de HumanMessage
        return response.content
    except Exception as e:
        logger.error(f"Error al obtener la respuesta desde Langchain: {e}")
        raise HTTPException(status_code=500, detail="Error al procesar la solicitud.")


@router.post("/chat_test/")
async def chat(message: Message):
    response = get_response_langchain(message.content)
    return {"response": response}
