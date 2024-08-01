from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chainlit as cl
import os
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

DB_FAISS_PATH = 'vectorstores/db_faiss'
DATA_PATH = 'data/PDF/Approvals/'
GROQ_MODEL_NAME='gemma-7b-it'

def LoadDocuments(file_type):
    match file_type:
        case 'PDF':
            glob = '*.pdf'
            loader_cls = PyPDFLoader
            recursive = False
        case 'HTML':
            glob = '**/*.html'
            loader_cls = BSHTMLLoader
            recursive = True
        
    return DirectoryLoader(path=DATA_PATH, glob=glob, loader_cls=loader_cls, recursive=recursive)

def load_documents():
    loader = LoadDocuments('PDF')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(texts)
    return texts

def create_database(embeddings):
    if not os.path.exists(DB_FAISS_PATH):
        texts = load_documents()
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(DB_FAISS_PATH)

def load_database(embeddings):
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db
    

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def on_start():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    create_database(embeddings)
    db = load_database(embeddings)
    retriever = db.as_retriever()
    return retriever

# QA Model Function
async def qa_bot(query, retriever):
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatGroq(model_name=GROQ_MODEL_NAME, temperature=0)
    rag_chain = (
        {"context": retriever | format_docs , "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    yield rag_chain.stream(query)

# Chainlit code
@cl.on_chat_start
async def start():
    retriever = on_start()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Meta-GPT. What is your query?"
    await msg.update()
    cl.user_session.set("retriever", retriever)

@cl.on_message
async def main(message: cl.Message):
    retriever = cl.user_session.get("retriever") 
    
    msg = cl.Message(content="")
    await msg.send()
    
    async for stream in qa_bot(message.content, retriever):
        for part in stream:
            await msg.stream_token(part)
                
    await msg.update()

# action categories related to approvals
# add comments to approval requests