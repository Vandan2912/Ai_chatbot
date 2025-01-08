import logging
from typing import Optional, List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_retrieval_qa_chain(
    vector_store_path: str, 
    model: str = 'llama3.1:8b',
    embedding_model: Optional[str] = None,
    search_kwargs: dict = {'k': 3}
):
    """
    Create a retrieval QA chain with Ollama model and FAISS vector store.
    
    :param vector_store_path: Path to the persistent FAISS vector store
    :param model: Ollama model to use
    :param embedding_model: Optional embedding model (defaults to same as chat model)
    :param search_kwargs: Retriever search configuration
    :return: Configured retrieval QA chain
    """
    try:
        logging.info("Initializing Retrieval QA chain...")
        
        # Use the chat model as embedding model if not specified
        embedding_model = embedding_model or model
        logging.info(f"Using embedding model: {embedding_model}")
        
        # Initialize Ollama Embeddings
        logging.info("Initializing Ollama embeddings...")
        embedding = OllamaEmbeddings(model=embedding_model)

        # Load existing vector store
        logging.info(f"Loading FAISS vector store from path: {vector_store_path}")
        vectorstore = FAISS.load_local(
            vector_store_path, 
            embedding, 
            allow_dangerous_deserialization=True
        )
        logging.info("FAISS vector store successfully loaded.")
        
        # Configure retriever
        logging.info(f"Configuring retriever with search parameters: {search_kwargs}")
        retriever = vectorstore.as_retriever(
            search_kwargs=search_kwargs  # Retrieve top k most relevant documents
        )
        
        # Define prompt template
        logging.info("Defining prompt template...")
        prompt_template = ChatPromptTemplate.from_template("""
        You are an AI assistant tasked with answering questions accurately and concisely based solely on the provided context. If the required information is not found in the context, respond only with "I don't know."

        ### Context:
        {context}

        ### Question:
        {question}

        ### Answer:
        """)
        logging.info("Prompt template successfully created.")
        
        # Function to format documents
        def format_docs(docs: List[Document]) -> str:
            """Format retrieved documents into a single context string."""
            logging.info(f"Formatting {len(docs)} retrieved documents into context...")
            formatted_docs = "\n\n".join(doc.page_content for doc in docs)
            logging.info("Document formatting complete.")
            return formatted_docs
        
        # Create Ollama LLM
        logging.info(f"Initializing Ollama LLM with model: {model}")
        llm = Ollama(model=model, temperature=0)
        logging.info("Ollama LLM initialized successfully.")
        
        # Construct the chain
        logging.info("Constructing retrieval QA chain...")
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )
        logging.info("Retrieval QA chain successfully created.")
        
        return chain

    except Exception as e:
        logging.error(f"Error while creating the retrieval QA chain: {e}")
        raise

def get_answer(
    question: str, 
    vector_store_path: str, 
    model: str = 'llama3.1:8b',
    search_kwargs: dict = {'k': 3}
) -> str:
    """
    Generate an answer to a question based on a PDF's vector store using Ollama and FAISS.
    
    :param question: The question to ask
    :param vector_store_path: Path to the persistent FAISS vector store
    :param model: Ollama model to use
    :param search_kwargs: Retriever search configuration
    :return: Generated answer
    """
    try:
        logging.info(f"Received question: {question}")
        logging.info("Initializing retrieval QA chain...")
        
        # Create the retrieval QA chain
        chain = create_retrieval_qa_chain(
            vector_store_path=vector_store_path, 
            model=model,
            search_kwargs=search_kwargs
        )
        logging.info("Retrieval QA chain initialized successfully.")
        
        # Invoke the chain to get the answer
        logging.info("Invoking the chain to generate an answer...")
        answer = chain.invoke(question)
        logging.info("Answer successfully generated.")
        return answer

    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        return f"An error occurred while processing the question: {str(e)}"

# # Initialize LLM
# llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
# embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# def get_answer(question: str, vector_store_path: str) -> str:
#     """
#     Generate an answer to a question based on a PDF's vector store
#     """
#     # Load vector store
#     vectorstore = Chroma(persist_directory=vector_store_path, embedding_function=embedding)

#     # Create QA chain
#     retriever = vectorstore.as_retriever()
#     qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

#     # Get answer
#     return qa_chain.run(question)
