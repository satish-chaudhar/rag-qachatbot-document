import os
import fitz  # PyMuPDF
import streamlit as st
from docx import Document as DocxDocument
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangDoc
from docx.opc.constants import RELATIONSHIP_TYPE as RT

# Load OpenAI Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# --- Helper Functions ---

def extract_text(file):
    ext = os.path.splitext(file.name)[-1].lower()
    if ext == ".pdf":
        return extract_pdf(file)
    elif ext == ".docx":
        return extract_docx(file)
    elif ext == ".txt":
        return file.read().decode("utf-8")
    else:
        return ""

def extract_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()

            for link in page.get_links():
                if 'uri' in link:
                    uri = link['uri']
                    text += f"\n[LINK] {uri}\n"
    return text

def extract_docx(file):
    doc = DocxDocument(file)
    full_text = []

    for para in doc.paragraphs:
        full_text.append(para.text)

    rels = doc.part.rels
    for rel in rels.values():
        if rel.reltype == RT.HYPERLINK:
            full_text.append(f"[LINK] {rel.target_ref}")

    return "\n".join(full_text)

def create_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = splitter.split_documents([LangDoc(page_content=text)])
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# --- Streamlit App ---

st.set_page_config(page_title="RAG Chatbot with FAISS", layout="centered")
st.title("🤖 Chat with Your Documents (FAISS-based)")

uploaded_files = st.file_uploader("Upload .txt, .pdf, .docx files", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing documents..."):
        full_text = ""
        for file in uploaded_files:
            full_text += extract_text(file) + "\n"

        vectorstore = create_vectorstore(full_text)
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        st.success("Upload complete! Ask your questions below.")

        question = st.text_input("Ask a question:")
        if st.button("Get Answer"):
            if question:
                with st.spinner("Thinking..."):
                    answer = qa_chain.run(question)
                    st.markdown(f"**Answer:** {answer}")
            else:
                st.warning("Enter a question to get started.")
