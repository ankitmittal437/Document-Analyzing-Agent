from langchain_azure_ai.embeddings import AzureAIEmbeddingsModel
from langchain_community.vectorstores import AzureSearch
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, UnstructuredImageLoader
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_community.retrievers import AzureAISearchRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from typing import List, Dict, Any
from dotenv import load_dotenv
from datetime import datetime

import tempfile
import os
import streamlit as st
import uuid

load_dotenv()

#Setting up the environment variables
os.environ["AZURE_INFERENCE_CREDENTIAL"] = os.getenv("Azure_AI_Foundry_Key")
os.environ["AZURE_INFERENCE_ENDPOINT"] = os.getenv("Azure_AI_Foundry_Endpoint")

embed_model = AzureAIEmbeddingsModel(
    model="text-embedding-3-large"
)

vector_store: AzureSearch = AzureSearch(
    embedding_function=embed_model.embed_query,
    azure_search_endpoint=os.getenv("AZURE_AI_SEARCH_SERVICE_NAME"),
    azure_search_key=os.getenv("AZURE_AI_SEARCH_API_KEY"),
    index_name="document-analyzer",
)

llm = AzureAIChatCompletionsModel(
    model="gpt-4o-mini",
    temprature=0.8
)

retriever = AzureAISearchRetriever(
    index_name="document-analyzer",
    content_key="content"
)

def process_url(url):
    loader = WebBaseLoader(url)

    docs = loader.load()
    doc_chunks = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200).split_documents(docs)

    return doc_chunks

def process_files(file_path: str, file_type: str, filename: str) -> List[Dict]:

    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "image":
        loader = UnstructuredImageLoader(file_path)

    docs = loader.load()
    doc_chunks = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200).split_documents(docs)

    documents = []
    for i, chunk in enumerate(doc_chunks):
        doc_id = str(uuid.uuid4())
        documents.append({
            "id": doc_id,
            "content": chunk,
            "metadata": {
                "filename": filename,
                "file_type": file_type,
                "chunk_index": i,
                "total_chunks": len(doc_chunks),
                "timestamp": datetime.now().isoformat()
            }
        })
    
    return doc_chunks

def store_embeddings(documents):
    if not documents:
        return

    try:
        vector_store.add_documents(documents = documents)
        st.success(f"Successfully stored {len(documents)} document chunks!")

    except Exception as e:
        st.error(f"Error storing documents: {e}")

def get_response(user_input):
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate respone based on the question
        <context>
        {context}
        <context>
        Question:{input}
        """
    )

    document_chain = create_stuff_documents_chain(llm,prompt)
    retrieval_chain = create_retrieval_chain(retriever,document_chain)

    response = retrieval_chain.invoke({'input':user_input})

    return response

    
def main():
    # Setting the page title details
    st.set_page_config(
        page_title="Document Chat System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    #Sidebar
    with st.sidebar:
        #st.header("📊 System Status")
        #st.metric("Total Documents", 0)
        
        # Navigation
        st.header("🧭 Navigation")
        page = st.radio("Choose a page:", ["📄 Document Upload", "💬 Chat Interface"])
        
        if page == "📄 Document Upload":
            st.header("⚙️ Upload Settings")
            st.info("Upload documents to add them to the knowledge base")
            
        else:
            #st.header("⚙️ Chat Settings")
            #k_docs = st.slider("Documents to retrieve", 1, 10, 5)
            
            if st.button("🗑️ Clear Chat History"):
                if "messages" in st.session_state:
                    st.session_state.messages = []
                st.success("Chat history cleared!")

    # Main content
    if page == "📄 Document Upload":
        st.title("📄 Document Upload")
        st.markdown("Upload documents to store them in the vector database")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📁 File Upload")
            
            # File upload
            uploaded_files = st.file_uploader(
                "Choose files",
                accept_multiple_files=True,
                type=['pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg', 'gif', 'bmp']
            )
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    # Determine file type
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    
                    if file_extension in ['png', 'jpg', 'jpeg', 'gif', 'bmp']:
                        file_type = "image"
                    elif file_extension == 'pdf':
                        file_type = "pdf"
                    elif file_extension == 'docx':
                        file_type = "docx"
                    elif file_extension == 'txt':
                        file_type = "txt"
                    else:
                        st.warning(f"Unsupported file type: {file_extension}")
                        continue
                    
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Process document
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        documents = process_files(tmp_file_path, file_type, uploaded_file.name)

                        if documents:
                            store_embeddings(documents)
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
        
        with col2:
            st.header("🌐 URL Processing")
            
            # URL input
            url = st.text_input("Enter URL to process:")
            
            if st.button("Process URL") and url:
                with st.spinner(f"Processing URL: {url}..."):
                    documents = process_url(url)
                    if documents:
                        store_embeddings(documents)
        
    
    else:  # Chat Interface
        st.title("💬 Chat Interface")
        st.markdown("Ask questions about your uploaded documents")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Chat messages container
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if user_input := st.chat_input("Ask a question about your documents..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("Searching documents and generating answer..."):
                        result = get_response(user_input)
                        answer = result['answer']
                        st.markdown(answer)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})

                        # Show sources
                        with st.expander("📚 Sources"):
                            for context in result['context']:
                                st.markdown(context.metadata['metadata'])

if __name__ == "__main__":
    main()