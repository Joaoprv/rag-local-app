from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from typing import Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    # Modern text splitter with better parameters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )
    chunks = splitter.split_text(text)
    
    # Modern embeddings with optimized settings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create vector store with metadata
    vectorstore = FAISS.from_texts(
        chunks, 
        embedding=embeddings,
        metadatas=[{"chunk_id": i} for i in range(len(chunks))]
    )
    
    return vectorstore

def get_chat_chain(llm: Any, vectorstore: FAISS) -> RunnableWithMessageHistory:
    """Create modern conversational RAG chain using LCEL.
    
    Args:
        llm: Language model for generation
        vectorstore: FAISS vector store for retrieval
        
    Returns:
        Modern conversational chain with message history
    """
    # Setup retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    def get_session_history(session_id: str) -> ChatMessageHistory:
        """Get or create chat history for a session."""
        if session_id not in chat_sessions:
            chat_sessions[session_id] = ChatMessageHistory()
        return chat_sessions[session_id]
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
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
    
    return conversational_rag_chain