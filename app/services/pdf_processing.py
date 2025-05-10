# import os
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_ollama import OllamaEmbeddings
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Initialize Ollama Embeddings
# logging.info("Initializing Ollama Embeddings with model 'llama3.1:8b'")
# embedding = OllamaEmbeddings(model="llama3.1:8b")

# def process_pdf(pdf_path: str, pdf_name: str):
#     """
#     Process the uploaded PDF to create embeddings using FAISS, with detailed logs.
#     """
#     try:
#         logging.info(f"Starting PDF processing for file: {pdf_name}")

#         # # Define the vector store directory
#         vector_store_path = os.path.join("./vector_store", f"{pdf_name}_vectors")
#         logging.info(f"Vector store path set to: {vector_store_path}")

#         # # Create the directory for saving vector store, if not exists
#         # logging.info(f"Creating directory for vector store at: {vector_store_path}")
#         # create_directory(vector_store_path)
#         # logging.info(f"Directory created or already exists: {vector_store_path}")

#         # Load and split PDF into pages
#         logging.info(f"Loading PDF from path: {pdf_path}")
#         loader = PyPDFLoader(pdf_path)
#         pages = loader.load_and_split()
#         logging.info(f"Loaded PDF and split into {len(pages)} pages")

#         # Split text into manageable chunks
#         logging.info("Initializing RecursiveCharacterTextSplitter with chunk_size=1000 and chunk_overlap=200")
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         texts = text_splitter.split_documents(pages)
#         logging.info(f"Split PDF into {len(texts)} text chunks")

#         # Create FAISS vector store
#         logging.info("Creating FAISS vector store from text chunks")
#         vectorstore = FAISS.from_documents(documents=texts, embedding=embedding)
#         logging.info("FAISS vector store created successfully")

#         # Save the vector store locally
#         logging.info(f"Saving vector store locally at: {vector_store_path}")
#         vectorstore.save_local(vector_store_path)
#         logging.info("Vector store saved successfully")

#         return vectorstore

#     except Exception as e:
#         logging.error(f"An error occurred while processing the PDF: {e}")
#         raise


import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from app.services.embedding_service import get_embedding_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_langchain_embedding():
    """
    Get the LangChain HuggingFace embeddings wrapper
    """
    model = get_embedding_model()
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': model.device},
        encode_kwargs={'normalize_embeddings': True}
    )

def process_pdf(pdf_path: str, pdf_name: str):
    """
    Process the uploaded PDF to create embeddings using FAISS with HuggingFace Sentence Transformers.
    Args:
        pdf_path (str): Path to the PDF file
        pdf_name (str): Name of the PDF file for naming the vector store
    Returns:
        FAISS: The vector store containing the document embeddings
    """
    try:
        logger.info(f"📄 Starting PDF processing for file: {pdf_name}")
        
        # Get the embedding model
        embedding = get_langchain_embedding()
        
        # Define the vector store directory
        vector_store_path = os.path.join("./vector_store", f"{pdf_name}_vectors")
        logger.info(f"📁 Vector store path set to: {vector_store_path}")
        
        # Load and split PDF into pages
        logger.info(f"📖 Loading PDF from path: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        logger.info(f"📑 Loaded PDF and split into {len(pages)} pages")
        
        # Split text into manageable chunks
        logger.info("✂️ Initializing RecursiveCharacterTextSplitter with chunk_size=1000 and chunk_overlap=200")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(pages)
        logger.info(f"📝 Split PDF into {len(texts)} text chunks")
        
        # Create FAISS vector store
        logger.info("🔍 Creating FAISS vector store from text chunks using HuggingFace embeddings")
        vectorstore = FAISS.from_documents(
            documents=texts,
            embedding=embedding
        )
        logger.info("✅ FAISS vector store created successfully")
        
        # Save the vector store locally
        logger.info(f"💾 Saving vector store locally at: {vector_store_path}")
        vectorstore.save_local(vector_store_path)
        logger.info("✅ Vector store saved successfully")
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"❌ An error occurred while processing the PDF: {e}")
        raise