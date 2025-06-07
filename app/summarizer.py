from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_summarizer_chain(llm, language="português") -> Runnable:
    """Create a summarization chain with improved prompting."""
    
    if language.lower() == "português":
        template = """Você é um assistente especializado em resumir documentos. 
        
Por favor, resuma o seguinte texto de forma clara e concisa, destacando os pontos principais:

Texto: {text}

Resumo em português:"""
    else:  # English
        template = """You are a document summarization specialist.

Please provide a clear and concise summary of the following text, highlighting the main points:

Text: {text}

Summary:"""
    
    prompt = PromptTemplate(
        input_variables=["text"],
        template=template
    )
    
    return prompt | llm


def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Split text into manageable chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return text_splitter.split_text(text)


def summarize(text, llm, language="português", max_chunk_size=2000):
    """
    Summarize text with support for long documents.
    
    Args:
        text (str): Text to summarize
        llm: Language model to use
        language (str): Language for the summary
        max_chunk_size (int): Maximum size of text chunks
    
    Returns:
        str: Generated summary
    """
    
    if not text.strip():
        return "No text provided for summarization."
    
    try:
        # If text is short enough, summarize directly
        if len(text) <= max_chunk_size:
            chain = get_summarizer_chain(llm, language)
            result = chain.invoke({"text": text})
            
            # Handle different response formats
            if isinstance(result, dict):
                return result.get("text", str(result))
            return str(result)
        
        # For longer texts, chunk and summarize
        chunks = chunk_text(text, chunk_size=max_chunk_size//2)
        chain = get_summarizer_chain(llm, language)
        
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i+1}/{len(chunks)}")
            try:
                result = chain.invoke({"text": chunk})
                if isinstance(result, dict):
                    summary = result.get("text", str(result))
                else:
                    summary = str(result)
                chunk_summaries.append(summary)
            except Exception as e:
                print(f"Error summarizing chunk {i+1}: {e}")
                chunk_summaries.append(f"[Error processing chunk {i+1}]")
        
        # Combine chunk summaries
        combined_summary = "\n\n".join(chunk_summaries)
        
        # If combined summary is still too long, summarize it again
        if len(combined_summary) > max_chunk_size:
            final_result = chain.invoke({"text": combined_summary})
            if isinstance(final_result, dict):
                return final_result.get("text", str(final_result))
            return str(final_result)
        
        return combined_summary
        
    except Exception as e:
        print(f"Error in summarization: {e}")
        return f"Error generating summary: {str(e)}"


def summarize_with_questions(text, llm, questions=None, language="português"):
    """
    Generate a summary focused on specific questions or aspects.
    
    Args:
        text (str): Text to summarize  
        llm: Language model to use
        questions (list): Specific questions to focus on
        language (str): Language for the summary
    
    Returns:
        str: Focused summary
    """
    
    if questions is None:
        questions = [
            "Quais são os principais tópicos abordados?",
            "Quais são as conclusões principais?", 
            "Existem dados ou estatísticas importantes?"
        ] if language == "português" else [
            "What are the main topics covered?",
            "What are the key conclusions?",
            "Are there important data or statistics?"
        ]
    
    questions_text = "\n".join(f"- {q}" for q in questions)
    
    if language.lower() == "português":
        template = f"""Baseado no texto fornecido, responda às seguintes perguntas para criar um resumo focado:

{questions_text}

Texto: {{text}}

Resumo estruturado:"""
    else:
        template = f"""Based on the provided text, answer the following questions to create a focused summary:

{questions_text}

Text: {{text}}

Structured summary:"""
    
    prompt = PromptTemplate(
        input_variables=["text"],
        template=template
    )
    
    chain = prompt | llm
    
    try:
        result = chain.invoke({"text": text})
        if isinstance(result, dict):
            return result.get("text", str(result))
        return str(result)
    except Exception as e:
        return f"Error generating focused summary: {str(e)}"