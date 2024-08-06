from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.llms.ctransformers import CTransformers
import chainlit as cl

MODEL_NAME = "TheBloke/Llama-2-7B-Chat-GGML"
SETTINGS = {
    'temperature': 0.01,
    'max_new_tokens': 600,
    'context_length': 1200
}
llm = CTransformers(
        model=MODEL_NAME,
        model_type="llama",
        config=SETTINGS
    )

@cl.on_chat_start
async def on_chat_start():
    model = llm
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions.",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
