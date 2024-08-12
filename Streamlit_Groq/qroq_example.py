import streamlit as st
from langchain_groq import ChatGroq
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough

GROQ_MODEL_NAME = 'gemma-7b-it'
st.title("🔎 metR Chatbot")
DB_FAISS_PATH = 'vectorstores/db_faiss'
DATA_PATH = 'data/PDF/Approvals/'

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

# Function to format the retrieved documents
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot. How can I help you?"}
    ]

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
if prompt := st.chat_input(placeholder="Write your query!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Set up the LLM and RAG chain
    llm = ChatGroq(model_name=GROQ_MODEL_NAME, temperature=0)
    search = DuckDuckGoSearchRun(name="Search")
    prompt_template = hub.pull("rlm/rag-prompt")  # Assuming this is the correct prompt
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(st.session_state.messages)
            
            if isinstance(response, list):
                response = ''.join(response)

            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
