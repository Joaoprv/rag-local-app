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
    
    # Create chat message history
    chat_history = ChatMessageHistory()
    
    def get_session_history(session_id: str) -> ChatMessageHistory:
        return chat_history
    
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
    
    # Modern LCEL chain
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
            "chat_history": RunnableLambda(lambda x: chat_history.messages),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Add conversation memory
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    
    return conversational_rag_chain

def main():
    """Test the modern RAG functions."""
    
    # Sample text for testing
    sample_text = """
    Machine learning is a subset of artificial intelligence that focuses on the development 
    of algorithms and statistical models that enable computers to improve their performance 
    on a specific task through experience. Deep learning, a subset of machine learning, 
    uses neural networks with multiple layers to model and understand complex patterns 
    in data. Natural language processing (NLP) is another important area that combines 
    computational linguistics with machine learning to help computers understand, interpret, 
    and generate human language. These technologies are revolutionizing industries from 
    healthcare to finance, enabling applications like medical diagnosis, fraud detection, 
    and automated customer service. Computer vision is another field that uses machine 
    learning algorithms to identify and analyze visual content in images and videos.
    """
    
    try:
        # Import required components for testing
        from langchain_huggingface import HuggingFacePipeline
        from transformers import pipeline
        
        logger.info("Building vector store...")
        # Build vector store
        vectorstore = build_vector_store(sample_text)
        logger.info("Vector store created successfully")
        
        logger.info("Setting up language model...")
        # Setup a simple LLM for testing
        text_generation_pipeline = pipeline(
            "text-generation",
            model="gpt2",
            max_length=150,
            temperature=0.7,
            pad_token_id=50256,
            truncation=True
        )
        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        logger.info("Language model loaded successfully")
        
        logger.info("Creating chat chain...")
        # Create conversational chain
        chat_chain = get_chat_chain(llm, vectorstore)
        logger.info("Chat chain created successfully")
        
        # Test questions
        test_questions = [
            "What is machine learning?",
            "How does deep learning relate to machine learning?", 
            "What are some applications mentioned?"
        ]
        
        logger.info("Starting conversational RAG system test")
        logger.info("="*60)
        
        session_config = {"configurable": {"session_id": "test_session"}}
        
        for i, question in enumerate(test_questions, 1):
            logger.info(f"Question {i}: {question}")
            
            try:
                response = chat_chain.invoke(
                    {"question": question},
                    config=session_config
                )
                logger.info(f"Answer: {response}")
                
            except Exception as e:
                logger.error(f"Error processing question: {e}")
        
        logger.info("="*60)
        logger.info("Testing completed successfully!")
        
    except ImportError as e:
        logger.error(f"Missing required packages: {e}")
        logger.error("Install with: pip install transformers torch langchain-huggingface")
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    main()