from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.ctransformers import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

app = FastAPI()

DATA_PATH = 'Aurigo/'
DB_FAISS_PATH = 'vectorstores/db_faiss'

class QueryRequest(BaseModel):
    query: str

# Global cache for the FAISS database
db_cache = None
llm_cache = None

@app.post("/create_database")
async def create_database():
    global db_cache
    try:
        loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={"device": "cpu"})
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
            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={"device": "cpu"})
            db_cache = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
            
        global llm_cache
        if llm_cache is None:
            llm_cache = CTransformers(
                model="TheBloke/Llama-2-7B-Chat-GGML",
                model_type="llama",
                config={
                    'temperature': 0.01,
                    'max_new_tokens': 600,
                    'context_length': 1200
                }
            )
        
        custom_prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Contexts: {context}

        Question: {question}
        Helpful Answer:"""
                
        qa_prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_cache,
            chain_type='stuff',
            retriever=db_cache.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': qa_prompt}
        )

        # Generate response
        response = qa_chain.invoke({'query': request.query})
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

        # custom_prompt_template = """
        #         ###Instructions:###
        #         You are an AI Integrated Chatbot. Your task is to give a relevant answer to the question asked by the User based on the context provided. The context will be pieces of information from a particular document. Please follow the following rules:
        #         1. For greetings like "hi", "hello", "how are you", and farewells like "bye", "good bye", always respond appropriately regardless of the contextAlways respond to greetings and good-byes.
        #         2. When the question is out of the scope of the context provided, just inform that it is out of your scope. If you don't know the answer, just say that you don't know gracefully; don't try to make up an answer. Answer the question given in a natural, human-like manner.
        #         3. Don't give improper or incorrect information or information that are not provided in the context. 
        #         4. Only return the helpful & correct answer below and nothing else. Verify the information before responding. 

        #         ###Context: {context}###
        #         ###Question: {question}###

        #         Helpful answer:
        #          """