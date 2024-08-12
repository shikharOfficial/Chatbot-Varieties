from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.ctransformers import CTransformers
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chainlit as cl
import os
import time
import torch

DB_FAISS_PATH = 'vectorstores/db_faiss'
DATA_PATH = 'data/PDF/Aurigo'
MODEL_NAME = "llama-2-7b-chat.Q8_0.gguf"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'temperature': 0.01,
    'max_new_tokens': 600,
    'context_length': 8000,
    'repetition_penalty': 1.1,
}
SETTINGS = {
    "model": MODEL_NAME,
    "model_type": "llama",
    "config": config
}
if device == "cuda":
    config['gpu_layers'] = 110
    SETTINGS["gpu_layers"] = 110
llm = CTransformers(**SETTINGS)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': device})
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

# Retrieval QA Chain
def retrieval_qa_chain(llm: CTransformers, prompt: PromptTemplate, db: FAISS):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 4}),
                                           chain_type_kwargs={'prompt': prompt},
                                           )
    return qa_chain

def create_database(embeddings: HuggingFaceEmbeddings):
    if not os.path.exists(DB_FAISS_PATH):
        loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(DB_FAISS_PATH)

# QA Model Function
def qa_bot():
    create_database(embeddings)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to metR-GPT. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    
    start_time = time.time() 
    
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    
    end_time = time.time() 
    duration = end_time - start_time
    
    answer += f"\nTime taken for query retrieval: {duration:.2f} seconds"

    await cl.Message(content=answer).send()
