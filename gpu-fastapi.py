from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.ctransformers import CTransformers
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import torch

app = FastAPI()

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
        
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class QueryRequest(BaseModel):
    query: str

db_cache = None

@app.post("/create_database")
async def create_database():
    global db_cache
    try:
        loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        db_cache = FAISS.from_documents(texts, embeddings)
        db_cache.save_local(DB_FAISS_PATH)
        return {"message": "Database created and saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_response")
async def generate_response(request: QueryRequest):
    try:
        global db_cache
        if db_cache is None:
            db_cache = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
            
        # prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
        # retriever = db_cache.as_retriever()

        # rag_chain = (
        #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
        #     | prompt
        #     | llm
        #     | StrOutputParser()
        # )
        # response = rag_chain.invoke(request.query)
        
        return {"response": db_cache}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)