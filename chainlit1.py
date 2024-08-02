from importlib.resources import contents
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.ctransformers import CTransformers
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chainlit as cl
import os
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

DB_FAISS_PATH = 'vectorstores/db_faiss'
DATA_PATH = 'data/PDF/Aurigo/'
MODEL_NAME = "TheBloke/Llama-2-7B-Chat-GGML"
SETTINGS = {
    'temperature': 0.01,
    'max_new_tokens': 600,
    'context_length': 1200
}

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def load_llm():
    llm = CTransformers(
        model=MODEL_NAME,
        model_type="llama",
        config=SETTINGS
    )
    return llm

def qa_bot(query: str, retriever):
    try:
        prompt = hub.pull("rlm/rag-prompt")
        
        llm = load_llm()
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("*"*50)
        print(rag_chain)
        print("*"*50)
        
        result = rag_chain.invoke(query)
        
        return result
        
    except Exception as e:
        print(f"Error in qa_bot: {e}")
        return str(e)

def create_database(embeddings):
    if not os.path.exists(DB_FAISS_PATH):
        loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(DB_FAISS_PATH)

def load_database(embeddings) -> FAISS:
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

def on_start():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    create_database(embeddings)
    db = load_database(embeddings)
    retriever = db.as_retriever(search_kwargs={'k': 2})
    
    # Extract file name
    files = os.listdir(DATA_PATH)
    pdf_file_name = [os.path.splitext(file)[0] for file in files if file.endswith('.pdf')][0]
    
    return retriever, pdf_file_name

@cl.on_chat_start
async def start():
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    retriever, pdf_file_name = on_start()
    msg.content = f"Hi, Welcome to metR-GPT. What is your query for {pdf_file_name}?"
    await msg.update()
    cl.user_session.set("retriever", retriever)

@cl.on_message
async def main(message: cl.Message):
    retriever = cl.user_session.get("retriever")  
    msg = cl.Message(content="")
    await msg.send()
    
    print(message.content, retriever)
    try:
        result = qa_bot(message.content, retriever)
        msg.content = result
        await msg.update()
    except Exception as e:
        print(f"Error in main: {e}")
        msg.content = f"Error: {e}"
        await msg.update()
