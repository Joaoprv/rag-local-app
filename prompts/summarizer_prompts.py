SUMMARIZER_PROMPTS = {
    "português": {
        "template": """Você é um assistente especializado em resumir documentos. 
        
Por favor, resuma o seguinte texto de forma clara e concisa, destacando os pontos principais:

Texto: {text}

Resumo em português:""",
        "input_variables": ["text"]
    },
    
    "english": {
        "template": """You are a document summarization specialist.

Please provide a clear and concise summary of the following text, highlighting the main points:

Text: {text}

Summary:""",
        "input_variables": ["text"]
    },
}

# Default language fallback
DEFAULT_LANGUAGE = "english"