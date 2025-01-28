 ## Local RAG System with DeepSeek and Streamlit

 This repository features a Local Retrieval-Augmented Generation (RAG) System powered by DeepSeek-Coder and Streamlit.
 It processes uploaded documents into a vector store and generates context-aware responses using a RAG pipeline.
  It includes rag_app.py for command-line use and streamlit_rag.py for a web interface. Utilizes Hugging Face Transformers,
  LangChain, and Chroma.
 ## Features
 - **Document Processing**: Upload and process text documents into a vector store using LangChain and Chroma.
  - **RAG Pipeline**: Generate responses using a Retrieval-Augmented Generation pipeline powered by DeepSeek-Coder.
  - **Interactive Interface**: A Streamlit-based web application for easy interaction with the RAG system.
  - **Local Execution**: Runs on both CPU and GPU, making it accessible for a wide range of hardware setups.
 
  ## Components
  1. **`rag_app.py`**: A standalone Python script for running the RAG system locally via the command line.
  2. **`streamlit_rag.py`**: A Streamlit-based web application for an interactive and user-friendly interface to the RAG system.
 
  ## Installation
 
  1. Clone the repository:
     ```bash
     git clone https://github.com/your-username/local-rag-system.git
     cd local-rag-system
     ```
 
  2. Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```
 
  3. (Optional) If you have a GPU, ensure you have the correct version of PyTorch installed for CUDA support:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
 
  ## Usage
 
  ### Using `rag_app.py`
  1. Place your text documents in the `documents` directory.
  2. Run the script:
     ```bash
     python rag_app.py
     ```
  3. Follow the prompts to ask questions and receive responses based on the uploaded documents.
 
  ### Using `streamlit_rag.py`
  1. Run the Streamlit application:
     ```bash
     streamlit run streamlit_rag.py
     ```
  2. Open the provided URL in your browser.
  3. Use the sidebar to initialize the system, upload documents, and configure generation settings.
  4. Interact with the chat interface to ask questions and view responses.
 
  ## Configuration
  - **Model**: The default model is `deepseek-ai/deepseek-coder-1.3b-base`. You can change this in the `LocalRAGSystem` class initialization.
  - **Embeddings**: The default embeddings model is `BAAI/bge-small-en-v1.5`. You can change this in the `LocalRAGSystem` class initialization.
  - **Generation Settings**: Adjust parameters like temperature, max_length, and num_docs in the Streamlit interface or directly in the code.
 
  ## Requirements
  - Python 3.8+
  - PyTorch
  - Transformers
  - LangChain
  - ChromaDB
  - Streamlit
 
  ## License
  This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
 
  ## Contributing
  Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
 
  ## Acknowledgments
  - **DeepSeek-Coder** for the language model.
  - **LangChain** for document processing and retrieval.
  - **Chroma** for vector storage.
  - **Streamlit** for the interactive web interface.
 
  Enjoy using the Local RAG System! 🚀
 
  ## Additional Notes
  - Ensure you have a `requirements.txt` file with all the necessary dependencies. Here s an example:
    ```plaintext
    torch
    transformers
    langchain
    chromadb
    streamlit
    sentence-transformers
    ```
  - Add a `.gitignore` file to exclude unnecessary files like `db/` (Chroma database) and `__pycache__/`.
 
  This setup will make your repository ready for sharing and collaboration! 🚀
