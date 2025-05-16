# import logging
# from typing import Optional, List
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_community.vectorstores import FAISS
# from langchain_ollama import OllamaEmbeddings
# from langchain_community.llms import Ollama
# from langchain_core.documents import Document

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def create_retrieval_qa_chain(
#     vector_store_path: str,
#     model: str = 'llama3.1:8b',
#     embedding_model: Optional[str] = None,
#     search_kwargs: dict = {'k': 3}
# ):
#     """
#     Create a retrieval QA chain with Ollama model and FAISS vector store.

#     :param vector_store_path: Path to the persistent FAISS vector store
#     :param model: Ollama model to use
#     :param embedding_model: Optional embedding model (defaults to same as chat model)
#     :param search_kwargs: Retriever search configuration
#     :return: Configured retrieval QA chain
#     """
#     try:
#         logging.info("Initializing Retrieval QA chain...")

#         # Use the chat model as embedding model if not specified
#         embedding_model = embedding_model or model
#         logging.info(f"Using embedding model: {embedding_model}")

#         # Initialize Ollama Embeddings
#         logging.info("Initializing Ollama embeddings...")
#         embedding = OllamaEmbeddings(model=embedding_model)

#         # Load existing vector store
#         logging.info(f"Loading FAISS vector store from path: {vector_store_path}")
#         vectorstore = FAISS.load_local(
#             vector_store_path,
#             embedding,
#             allow_dangerous_deserialization=True
#         )
#         logging.info("FAISS vector store successfully loaded.")

#         # Configure retriever
#         logging.info(f"Configuring retriever with search parameters: {search_kwargs}")
#         retriever = vectorstore.as_retriever(
#             search_kwargs=search_kwargs  # Retrieve top k most relevant documents
#         )

#         # Define prompt template
#         logging.info("Defining prompt template...")
#         prompt_template = ChatPromptTemplate.from_template("""
#         You are an AI assistant tasked with answering questions accurately and concisely based solely on the provided context. If the required information is not found in the context, respond only with "LLM unable to find answer"

#         ### Context:
#         {context}

#         {chat_history}

#         ### Question:
#         {question}

#         ### Answer:
#         """)
#         logging.info("Prompt template successfully created.")

#         # Function to format documents
#         def format_docs(docs: List[Document]) -> str:
#             """Format retrieved documents into a single context string."""
#             logging.info(f"Formatting {len(docs)} retrieved documents into context...")
#             formatted_docs = "\n\n".join(doc.page_content for doc in docs)
#             logging.info("Document formatting complete.")
#             return formatted_docs

#         # Create Ollama LLM
#         logging.info(f"Initializing Ollama LLM with model: {model}")
#         llm = Ollama(model=model, temperature=0)
#         logging.info("Ollama LLM initialized successfully.")

#         # Construct the chain
#         logging.info("Constructing retrieval QA chain...")
#         chain = (
#             {"context": retriever | format_docs, "question": RunnablePassthrough()}
#             | prompt_template
#             | llm
#             | StrOutputParser()
#         )
#         logging.info("Retrieval QA chain successfully created.")

#         return chain

#     except Exception as e:
#         logging.error(f"Error while creating the retrieval QA chain: {e}")
#         raise

# def get_answer(
#     question: str,
#     vector_store_path: str,
#     model: str = 'llama3.1:8b',
#     search_kwargs: dict = {'k': 3}
# ) -> str:
#     """
#     Generate an answer to a question based on a PDF's vector store using Ollama and FAISS.

#     :param question: The question to ask
#     :param vector_store_path: Path to the persistent FAISS vector store
#     :param model: Ollama model to use
#     :param search_kwargs: Retriever search configuration
#     :return: Generated answer
#     """
#     try:
#         logging.info(f"Received question: {question}")
#         logging.info("Initializing retrieval QA chain...")

#         # Create the retrieval QA chain
#         chain = create_retrieval_qa_chain(
#             vector_store_path=vector_store_path,
#             model=model,
#             search_kwargs=search_kwargs
#         )
#         logging.info("Retrieval QA chain initialized successfully.")

#         # Invoke the chain to get the answer
#         logging.info("Invoking the chain to generate an answer...")
#         answer = chain.invoke(question)
#         logging.info("Answer successfully generated.")
#         return answer

#     except Exception as e:
#         logging.error(f"Error generating answer: {e}")
#         return f"An error occurred while processing the question: {str(e)}"

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


import logging
import os
from typing import Optional, List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from app.utils.Constant import LLM_MODEL
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_retrieval_qa_chain(
    vector_store_path: str,
    model: str,
    embedding_model: str = "all-MiniLM-L6-v2",  # Default Sentence Transformer model
    search_kwargs: dict = {"k": 3},
    chat_history: list = None,
):
    """
    Create a retrieval QA chain with Ollama, OpenAI, or DeepSeek LLM and FAISS vector store using HuggingFace embeddings.
    The model type is determined by the model name.

    :param vector_store_path: Path to the persistent FAISS vector store
    :param model: Model name to use. The API is selected based on the model name:
                 - OpenAI models: ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"]
                 - DeepSeek models: ["deepseek-chat", "deepseek-coder"]
                 - All other model names will use Ollama
    :param embedding_model: Sentence Transformer model for embeddings
    :param search_kwargs: Retriever search configuration
    :return: Configured retrieval QA chain
    """
    try:
        logging.info("Initializing Retrieval QA chain...")

        # Initialize HuggingFace Embeddings
        logging.info(
            f"Initializing HuggingFace embeddings with model: {embedding_model}"
        )
        embedding = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},  # Use 'cuda' if you have GPU
            encode_kwargs={"normalize_embeddings": True},
        )

        # Load existing vector store
        logging.info(f"Loading FAISS vector store from path: {vector_store_path}")
        vectorstore = FAISS.load_local(
            vector_store_path, embedding, allow_dangerous_deserialization=True
        )
        logging.info("FAISS vector store successfully loaded.")

        # Configure retriever
        logging.info(f"Configuring retriever with search parameters: {search_kwargs}")
        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

        # Define prompt template
        logging.info("Defining prompt template...")
        prompt_template = ChatPromptTemplate.from_template(
            """
    You are an AI assistant representing our business. Respond to customer questions accurately and concisely, using only the information provided in the context.

    Guidelines:
    - Maintain a friendly, professional, and conversational tone
    - Use first-person plural pronouns ("we", "our", "us") to represent yourself as part of the business
    - Keep responses concise and directly address the customer's question
    - Format responses with appropriate paragraphs, bullet points, or numbered lists when needed for clarity
    - Include relevant specific details from the context (e.g., prices, dates, specifications)
    - NEVER invent information that isn't in the context
    - Always highlight positive aspects of our products/services when relevant without making exaggerated claims
    - When asked about competitor products or services, provide factual information only if it exists in the context
    - Never recommend competitor products or services over our own
    - Never make negative comments about our company, products, or services
    - Provide clear, straightforward answers in simple, easy-to-understand language
    - Be factual and to the point
    - Prioritize clarity and brevity
    - Tailor response length to question complexity
    - Use appropriate citations when referencing specific information from the context

    If the required information is not found in the context, respond only with "LLM unable to find answer".

    ### Context:
    {context}

    ### Chat History:
    {chat_history_text}

    ### Question:
    {question}

    Answer:
    """
        )

        logging.info("Prompt template successfully created.")

        # Function to format documents
        def format_docs(docs: List[Document]) -> str:
            """Format retrieved documents into a single context string."""
            logging.info(f"Formatting {len(docs)} retrieved documents into context...")
            formatted_docs = "\n\n".join(doc.page_content for doc in docs)
            logging.info("Document formatting complete.")
            return formatted_docs

        # Determine which API to use based on model name
        openai_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]
        deepseek_models = ["deepseek-chat"]
        
        if model in openai_models:
            # Use OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key must be set via OPENAI_API_KEY environment variable")
                
            logging.info(f"Initializing OpenAI LLM with model: {model}")
            llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)
            logging.info("OpenAI LLM initialized successfully.")
        elif model in deepseek_models:
            # Use DeepSeek via OpenAI-compatible API
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DeepSeek API key must be set via DEEPSEEK_API_KEY environment variable")
                
            logging.info(f"Initializing DeepSeek LLM with model: {model}")
            llm = ChatOpenAI(
                model=model,
                temperature=0,
                api_key=api_key,
                base_url="https://api.deepseek.com/v1"
            )
            logging.info("DeepSeek LLM initialized successfully.")
        else:
            # Use Ollama
            logging.info(f"Initializing Ollama LLM with model: {model}")
            llm = Ollama(model=model, temperature=0)
            logging.info("Ollama LLM initialized successfully.")

        # Format chat history if provided
        def format_chat_history(history):
            if not history or len(history) == 0:
                return {"chat_history_text": ""}
            
            formatted_history = "### Chat History:\n"
            for i, message in enumerate(history):
                formatted_history += f"Q{i+1}: {message.question}\n"
                formatted_history += f"A{i+1}: {message.answer}\n\n"
            
            return {"chat_history_text": formatted_history}
        
        # Construct the chain
        logging.info("Constructing retrieval QA chain...")
        chain = (
            {
                "context": retriever | format_docs, 
                "question": RunnablePassthrough(),
                "chat_history_text": lambda x: format_chat_history(chat_history).get("chat_history_text")
            }
            | prompt_template
            | llm
            | StrOutputParser()
        )
        logging.info("Retrieval QA chain successfully created.")
        logging.info(f"LLm model is {model}")

        return chain

    except Exception as e:
        logging.error(f"Error while creating the retrieval QA chain: {e}")
        raise


def get_answer(
    question: str, vector_store_path: str, model: str, search_kwargs: dict = {"k": 3}, chat_history: list = None
) -> str:
    """
    Generate an answer to a question based on a PDF's vector store using Ollama, OpenAI, or DeepSeek and FAISS.

    :param question: The question to ask
    :param vector_store_path: Path to the persistent FAISS vector store
    :param model: Model name to use. The API is selected based on the model name:
                 - OpenAI models: ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"]
                 - DeepSeek models: ["deepseek-chat"]
                 - All other model names will use Ollama
    :param search_kwargs: Retriever search configuration
    :param chat_history: List of previous question-answer pairs for conversation context
    :return: Generated answer
    """
    try:
        logging.info(f"Received question: {question}")
        logging.info("Initializing retrieval QA chain... ")

        # Create the retrieval QA chain
        chain = create_retrieval_qa_chain(
            vector_store_path=vector_store_path,
            model=model,
            search_kwargs=search_kwargs,
            chat_history=chat_history,
        )
        logging.info("Retrieval QA chain initialized successfully.")

        # Invoke the chain to get the answer
        logging.info("Invoking the chain to generate an answer...")
        answer = chain.invoke(question)
        logging.info("Answer successfully generated.")
        logging.info(f"LLm model is {model}")
        return answer

    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        return f"An error occurred while processing the question: {str(e)}"
