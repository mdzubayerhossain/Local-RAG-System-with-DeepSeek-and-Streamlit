import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import os
import logging
from typing import List, Dict, Any
import tempfile

class LocalRAGSystem:
    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-coder-1.3b-base",
        embeddings_model: str = "BAAI/bge-small-en-v1.5",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        with st.spinner("Loading language model..."):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True
            ).to(device)
        
        with st.spinner("Loading embedding model..."):
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embeddings_model,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        self.vector_store = None
        
    def process_uploaded_files(self, uploaded_files) -> None:
        """Process uploaded files and create vector store."""
        documents = []
        
        with st.spinner("Processing uploaded documents..."):
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name
                
                loader = TextLoader(temp_file_path)
                documents.extend(loader.load())
                os.unlink(temp_file_path)
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory="db"
            )
            st.success(f"Processed {len(splits)} document chunks")
        
    def generate_response(
        self,
        query: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        num_docs: int = 3
    ) -> Dict[str, Any]:
        """Generate a response using RAG pipeline."""
        if not self.vector_store:
            raise ValueError("No documents loaded. Please upload documents first.")
            
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": num_docs}
        )
        docs = retriever.get_relevant_documents(query)
        
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""Context information is below.
        ---------------------
        {context}
        ---------------------
        Given the context information and no other information, answer the following query:
        {query}
        
        Answer:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "query": query,
            "response": response,
            "context": context,
            "num_docs_retrieved": len(docs)
        }

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Local RAG System",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("Local RAG System with DeepSeek-R1")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for system initialization and file upload
    with st.sidebar:
        st.header("System Configuration")
        
        if st.button("Initialize System", key="init_system"):
            st.session_state.rag_system = LocalRAGSystem()
        
        if st.session_state.rag_system:
            st.success("System initialized!")
            
            uploaded_files = st.file_uploader(
                "Upload Documents",
                accept_multiple_files=True,
                type=['txt']
            )
            
            if uploaded_files:
                if st.button("Process Documents"):
                    st.session_state.rag_system.process_uploaded_files(uploaded_files)
        
        st.header("Generation Settings")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        max_length = st.slider("Max Length", 64, 1024, 512)
        num_docs = st.slider("Number of Retrieved Documents", 1, 5, 3)
    
    # Main chat interface
    st.header("Chat Interface")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.write("You:", message["content"])
            else:
                st.write("Assistant:", message["content"])
                if "context" in message:
                    with st.expander("View Retrieved Context"):
                        st.write(message["context"])
    
    # Query input
    if query := st.chat_input("Enter your question"):
        if not st.session_state.rag_system:
            st.error("Please initialize the system first!")
            return
            
        if not st.session_state.rag_system.vector_store:
            st.error("Please upload and process documents first!")
            return
        
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        try:
            with st.spinner("Generating response..."):
                result = st.session_state.rag_system.generate_response(
                    query,
                    temperature=temperature,
                    max_length=max_length,
                    num_docs=num_docs
                )
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["response"],
                "context": result["context"]
            })
            
            # Rerun to update chat display
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
