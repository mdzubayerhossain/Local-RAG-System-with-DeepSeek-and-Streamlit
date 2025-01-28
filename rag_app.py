import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import RetrievalQA
import os
import logging
from typing import List, Dict, Any

class LocalRAGSystem:
    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-coder-1.3b-base",
        embeddings_model: str = "BAAI/bge-small-en-v1.5",
        docs_dir: str = "documents",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.docs_dir = docs_dir
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load the LLM
        self.logger.info(f"Loading DeepSeek model on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(device)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize vector store
        self.vector_store = None
        
    def load_documents(self) -> None:
        """Load and process documents from the specified directory."""
        self.logger.info(f"Loading documents from {self.docs_dir}...")
        
        # Create loader for text files
        loader = DirectoryLoader(
            self.docs_dir,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="db"
        )
        self.logger.info(f"Processed {len(splits)} document chunks")
        
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
            raise ValueError("Documents haven't been loaded. Call load_documents() first.")
            
        # Retrieve relevant documents
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": num_docs}
        )
        docs = retriever.get_relevant_documents(query)
        
        # Construct prompt with retrieved context
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""Context information is below.
        ---------------------
        {context}
        ---------------------
        Given the context information and no other information, answer the following query:
        {query}
        
        Answer:"""
        
        # Generate response
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
        
    def save_vector_store(self) -> None:
        """Save the vector store to disk."""
        if self.vector_store:
            self.vector_store.persist()
            self.logger.info("Vector store saved to disk")
            
    def load_vector_store(self) -> None:
        """Load the vector store from disk."""
        if os.path.exists("db"):
            self.vector_store = Chroma(
                persist_directory="db",
                embedding_function=self.embeddings
            )
            self.logger.info("Vector store loaded from disk")
            
def main():
    # Initialize the RAG system
    rag = LocalRAGSystem()
    
    # Create documents directory if it doesn't exist
    if not os.path.exists("documents"):
        os.makedirs("documents")
        print("Created 'documents' directory. Please add your text files there.")
        return
        
    # Load documents and create vector store
    rag.load_documents()
    
    # Example usage
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        try:
            result = rag.generate_response(query)
            print("\nResponse:", result["response"])
            print("\nBased on", result["num_docs_retrieved"], "relevant documents")
        except Exception as e:
            print(f"Error: {str(e)}")
            
    # Save vector store before exiting
    rag.save_vector_store()

if __name__ == "__main__":
    main()
