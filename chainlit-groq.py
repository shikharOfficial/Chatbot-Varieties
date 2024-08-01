from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chainlit as cl
from groq import Groq
import os
from typing import Tuple

DB_FAISS_PATH = 'vectorstores/db_faiss'
DATA_PATH = 'data/HTML/APPROVALS/'
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

custom_prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Contexts: {context}

        Question: {question}
        Helpful Answer:"""
        
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

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def load_documents():
    loader = LoadDocuments('HTML')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    return texts

def create_database(embeddings):
    if not os.path.exists(DB_FAISS_PATH):
        texts = load_documents()
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(DB_FAISS_PATH)

def load_database(embeddings):
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

def on_start():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    create_database(embeddings)
    db = load_database(embeddings)
    return db

def init_groq(query, db: FAISS) -> Tuple[Groq, str]:
    retrieved_docs = db.similarity_search(query=query, k=4)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    client = Groq(api_key=GROQ_API_KEY)
    sys_prompt = f"""
    Instructions:
    - Be helpful and answer questions concisely. If you don't know the answer, say 'I don't know'
    - Utilize the context provided for accurate and specific information. If you can't find any information from the context, just respond you don't know as not provided in the context.
    - Just respond with the answers available in the context.
    - Never show sources from where you extracted final answer in your response.
    - Cite your sources
    Context: {context}
    """
    
    return client, sys_prompt

async def stream_response(client: Groq, sys_prompt: str, query: str):
    stream = client.chat.completions.create(
        messages=[
            { 
                'role': 'system',
                'content': sys_prompt
            },
            {
                'role': 'user',
                'content': query
            }
        ],
        model="gemma-7b-it",
        temperature=0.1,
        stream=True
    )
    
    yield stream

@cl.on_chat_start
async def start():
    database = on_start()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to MetR-GPT. What is your query?"
    await msg.update()
    cl.user_session.set("database", database)

@cl.on_message
async def main(message: cl.Message):
    database = cl.user_session.get("database") 
    query = message.content
    
    msg = cl.Message(content="")
    await msg.send()

    client, sys_prompt = init_groq(query, database)

    async for stream in stream_response(client, sys_prompt, query):
        for part in stream:
            if token := part.choices[0].delta.content or "":
                await msg.stream_token(token)
            
    await msg.update()
