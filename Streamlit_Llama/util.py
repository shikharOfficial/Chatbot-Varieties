from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import time
from langchain_community.llms.ctransformers import CTransformers

DB_FAISS_PATH = 'vectorstores/db_faiss'

def load_llm():
    llm = CTransformers(
                model="TheBloke/Llama-2-7B-Chat-GGML",
                model_type="llama",
                config={
                    'temperature': 0.01,
                    'max_new_tokens': 600,
                    'context_length': 8000
                }
            )
    return llm

# Read PDF data
def read_pdf_data(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Split data into chunks
def split_data(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(text)
    return text_chunks


def get_embedding_function():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    return embeddings


# Create vectorstore
def create_vectorstore(embeddings, pdf_docs):
    raw_text = read_pdf_data(pdf_docs)  # Get PDF text
    text_chunks = split_data(raw_text)  # Get the text chunks
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local(DB_FAISS_PATH)
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_llm_response(llm, prompt, question, embeddings):
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    start_time = time.time()
    retriever = db.as_retriever()
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke(question)
    end_time = time.time() 
    duration = end_time - start_time
    
    response += f"\nTime taken for query retrieval: {duration:.2f} seconds"
    
    return response