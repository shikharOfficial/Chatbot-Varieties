from util import *
from streamlit_option_menu import option_menu
from langchain import hub
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Doc Chat", page_icon=":robot_face:", layout="centered")

# --- SETUP SESSION STATE VARIABLES ---
if "database" not in st.session_state:
    llm = load_llm()
    embeddings = get_embedding_function()
if "vector_store" not in st.session_state:
    st.session_state.vector_store = False
if "response" not in st.session_state:
    st.session_state.response = None
if "prompt_activation" not in st.session_state:
    st.session_state.prompt_activation = True
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None
if "prompt" not in st.session_state:
    st.session_state.prompt = False

# --- MAIN PAGE CONFIGURATION ---
st.title("Doc Chat :robot_face:")
st.write("*Interrogate Documents :books:, Ignite Insights: AI at Your Service*")
st.write(':blue[***Powered by Llama AI Inference Technology***]')

# ---- NAVIGATION MENU -----
selected = option_menu(
    menu_title=None,
    options=["Doc Chat", "About"],
    icons=["robot", "bi-file-text-fill", "app"],  # https://icons.getbootstrap.com
    orientation="horizontal",
)

prompt = hub.pull("rlm/rag-prompt")

# ----- SETUP Doc Chat MENU ------
if selected == "Doc Chat":
    st.subheader("Upload PDF(s)")
    pdf_docs = st.file_uploader("Upload your PDFs", type='pdf', accept_multiple_files=True,
                                disabled=not st.session_state.prompt_activation, label_visibility='collapsed')
    process = st.button("Process", type="primary", key="process", disabled=not pdf_docs)

    if process:
        with st.spinner("Processing ..."):
            st.session_state.vector_store = create_vectorstore(embeddings, pdf_docs)
            st.session_state.prompt = True
            st.success('Database is ready')

    st.divider()

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    container = st.container(border=True)
    if question := st.chat_input(placeholder='Enter your question related to uploaded document',
                                 disabled=not st.session_state.prompt):
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner('Processing...'):
            st.session_state.response = get_llm_response(llm, prompt, question, embeddings)
            st.session_state.messages.append({"role": "assistant", "content": st.session_state.response})
            st.chat_message("assistant").write(st.session_state.response)

# ----- SETUP ABOUT MENU ------
if selected == "About":
    with st.expander("About this App"):
        st.markdown(''' This app allows you to chat with your PDF documents.''')
    with st.expander("Which Large Language models are supported by this App?"):
        st.markdown(''' This app supports the following LLM:
    - Chat Models  
        - TheBloke/Llama-2-7B-Chat-GGUF
        ''')

    with st.expander("Which library is used for vectorstore?"):
        st.markdown(''' This app supports the FAISS for AI similarity search and vectorstore:
        ''')