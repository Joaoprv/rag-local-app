import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from typing import Any

logger = logging.getLogger(__name__)

# Global chat history storage
chat_sessions = {}

def build_vector_store(text: str) -> FAISS:
    """Build FAISS vector store from text using modern components.
    
    Args:
        text: Input text to be indexed
        
    Returns:
        FAISS vector store
    """
    logger.info("Building vector store from text")
    
    # Modern text splitter with better parameters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )
    chunks = splitter.split_text(text)
    logger.info(f"Text split into {len(chunks)} chunks")
    
    # Modern embeddings with optimized settings
    logger.info("Loading embeddings model")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create vector store with metadata
    logger.info("Creating FAISS vector store")
    vectorstore = FAISS.from_texts(
        chunks, 
        embedding=embeddings,
        metadatas=[{"chunk_id": i} for i in range(len(chunks))]
    )
    
    logger.info("Vector store created successfully")
    return vectorstore

def get_chat_chain(llm: Any, vectorstore: FAISS) -> RunnableWithMessageHistory:
    """Create modern conversational RAG chain using LCEL.
    
    Args:
        llm: Language model for generation
        vectorstore: FAISS vector store for retrieval
        
    Returns:
        Modern conversational chain with message history
    """
    logger.info("Creating conversational RAG chain")
    
    # Setup retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    logger.debug("Retriever configured with k=4 similarity search")
    
    def get_session_history(session_id: str) -> ChatMessageHistory:
        """Get or create chat history for a session."""
        if session_id not in chat_sessions:
            logger.debug(f"Creating new chat session: {session_id}")
            chat_sessions[session_id] = ChatMessageHistory()
        return chat_sessions[session_id]
    
    def format_docs(docs):
        formatted = "\n\n".join(doc.page_content for doc in docs)
        logger.debug(f"Formatted {len(docs)} documents for context")
        return formatted
    
    # Modern prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. Use the following context to answer questions accurately and concisely.

Context: {context}

If you cannot find the answer in the context, please say so clearly."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    rag_chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
            "chat_history": lambda x: [], 
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    
    logger.info("Conversational RAG chain created successfully")
    return conversational_rag_chain