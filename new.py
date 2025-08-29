import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

API_KEY = "AIzaSyB4NNbflQD3Hf7MCCO9TmGWh3Gm3hoR1x0"

try:
    genai.configure(api_key=API_KEY)
    print("Google Generative AI configured successfully.")
except Exception as e:
    st.error(f"Failed to configure Google Generative AI: {e}")
    st.stop()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        st.warning("No text to process. Please upload a PDF with text content.")
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "The answer is not available in the context." Do not provide a wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def handle_user_input(user_question):
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        st.warning("Please upload and process your PDFs first.")
        return "Please upload and process your PDFs first."

    vector_store = st.session_state.vector_store
    try:
        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        return f"An error occurred: {e}"

def main():
    st.set_page_config(page_title="Chat with PDF using Gemini", page_icon="📄")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload your PDFs and I'll help you find answers within them."}]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    st.header("Chat with your PDFs using Gemini 💬")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = handle_user_input(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

    with st.sidebar:
        st.title("Settings")
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDF files here", accept_multiple_files=True, type="pdf")
        
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing documents... this may take a moment."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = get_vector_store(text_chunks)
                    
                    if st.session_state.vector_store:
                        st.success("Processing complete! You can now ask questions.")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()