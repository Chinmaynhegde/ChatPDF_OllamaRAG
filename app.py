import streamlit as st
from langchain.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

def process_pdf(file):
    # Save the file to a temporary location
    with open("uploaded_file.pdf", "wb") as f:
        f.write(file.getbuffer())

    # Load PDF
    loader = UnstructuredPDFLoader(file_path="uploaded_file.pdf")
    data = loader.load()

    # Split and chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)

    # Add to vector database
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
        collection_name="local-rag"
    )

    return vector_db

def setup_retriever(vector_db):
    # LLM from Ollama
    local_model = "mistral"
    llm = ChatOllama(model=local_model)

    # Retrieval setup
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    # RAG prompt
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Streamlit app
st.title("RAG System with Ollama")
st.write("Upload a PDF document and ask a question based on its content.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    vector_db = process_pdf(uploaded_file)
    chain = setup_retriever(vector_db)
    
    question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if question:
            answer = chain.invoke(question)
            st.write("### Answer")
            st.write(answer)
        else:
            st.write("Please enter a question.")
