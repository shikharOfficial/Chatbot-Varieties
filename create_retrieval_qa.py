from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.ctransformers import CTransformers
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.runnables import Runnable
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chainlit as cl
import os

DB_FAISS_PATH = 'vectorstores/db_faiss'
DATA_PATH = 'data/PDF/Aurigo'
MODEL_NAME = "llama-2-7b-chat.Q8_0.gguf"
SETTINGS = {
    'temperature': 0.01,
    'max_new_tokens': 600,
    'context_length': 8000
}
llm = CTransformers(
        model=MODEL_NAME,
        model_type="llama",
        config=SETTINGS 
    )
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
custom_prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Contexts: {context}

        Question: {input}
        Helpful Answer:"""

def set_custom_prompt():
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    return prompt


def retrieval_qa_chain(llm: CTransformers, prompt: ChatPromptTemplate, db: FAISS):
    qa_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever()
    
    retrieval_chain = create_retrieval_chain(retriever, qa_chain)
    
    return retrieval_chain


def create_database(embeddings: HuggingFaceEmbeddings):
    if not os.path.exists(DB_FAISS_PATH):
        loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(DB_FAISS_PATH)
        

#QA Model Function
def qa_bot() -> Runnable:
    create_database(embeddings)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    qa_prompt = set_custom_prompt()
    retrieval_chain = retrieval_qa_chain(llm, qa_prompt, db)
    
    return retrieval_chain

#chainlit code
@cl.on_chat_start
async def start():
    runnable = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to MetR-GPT. What is your query?"
    await msg.update()

    cl.user_session.set("runnable", runnable)

@cl.on_message
async def main(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    
    msg = cl.Message(content="")

    res = runnable.invoke({"input": message.content})
    
    await msg.send(content=res['answer'])

