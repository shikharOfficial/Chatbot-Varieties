from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chainlit as cl
from groq import Groq
import os

DB_FAISS_PATH = 'vectorstores/db_faiss'
DATA_PATH = 'data/'
GROQ_API_KEY= os.environ.get('GROQ_API_KEY')

custom_prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Contexts: {context}

        Question: {question}
        Helpful Answer:"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
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

# Groq QA Chain
def groq_qa_chain(query, context):
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
        temperature=0.1
    )
    
    return stream.choices[0].message.content

def on_start():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    create_database(embeddings)
    db = load_database(embeddings)
    return db

# QA Model Function
def qa_bot(query, db: FAISS):
    retrieved_docs = db.similarity_search(query=query, k=4)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    print(context)

    response = groq_qa_chain(query, context)
    return response

# Chainlit code
@cl.on_chat_start
async def start():
    database = on_start()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Meta-GPT. What is your query?"
    await msg.update()
    cl.user_session.set("database", database)

@cl.on_message
async def main(message: cl.Message):
    database = cl.user_session.get("database") 
    response = qa_bot(message.content, database)
    await cl.Message(content=response).send()
