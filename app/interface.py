import gradio as gr
from .pdf_loader import extract_text_from_pdf
from .summarizer import summarize
from .chat_rag import build_vector_store, get_chat_chain
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Global variables to store state
current_llm = None
chat_chain = None

def load_llm(model_name="tiiuae/falcon-rw-1b"):
    """Load the specified language model with better error handling."""
    try:
        print(f"Loading model: {model_name}")
        
        if model_name == "tiiuae/falcon-rw-1b":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            pipe = pipeline(
                "text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            return HuggingFacePipeline(pipeline=pipe)
            
        elif model_name == "meta-llama/llama-2-7b-chat-hf":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            pipe = pipeline(
                "text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            return HuggingFacePipeline(pipeline=pipe)
        else:
            raise ValueError(f"Invalid model: {model_name}")
            
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        return None

def update_llm(model_name):
    """Update the global LLM with better state management."""
    global current_llm, chat_chain
    
    try:
        new_llm = load_llm(model_name)
        if new_llm is not None:
            current_llm = new_llm
            # Reset chat chain when model changes
            chat_chain = None
            return f"✅ Model '{model_name}' loaded successfully!"
        else:
            return f"❌ Failed to load model '{model_name}'"
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"

def process_pdf(file):
    """Process PDF with better error handling and state management."""
    global current_llm, chat_chain
    
    if current_llm is None:
        return "❌ Please load a model first!"
    
    if file is None:
        return "❌ Please upload a PDF file!"
    
    try:
        print(f"Processing PDF: {file.name}")
        
        # Extract text from PDF
        text = extract_text_from_pdf(file.name)
        if not text.strip():
            return "❌ No text could be extracted from the PDF!"
        
        # Generate summary
        summary = summarize(text, current_llm)
        
        # Build vector store and chat chain
        vectorstore = build_vector_store(text)
        chat_chain = get_chat_chain(current_llm, vectorstore)
        
        return f"✅ PDF processed successfully!\n\n📄 Summary:\n{summary}"
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return f"❌ Error processing PDF: {str(e)}"

def chat_with_pdf(user_input):
    """Chat with PDF with better error handling."""
    global chat_chain
    
    if not user_input.strip():
        return "Please enter a question."
    
    if chat_chain is None:
        return "❌ Please upload and process a PDF first!"
    
    try:
        response = chat_chain.invoke({"question": user_input})
        # Handle different response formats
        if isinstance(response, dict):
            return response.get("answer", response.get("result", str(response)))
        return str(response)
        
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return f"❌ Error generating response: {str(e)}"

# Initialize with default model
print("Initializing with default model...")
current_llm = load_llm()

# Create Gradio interface
demo = gr.Blocks(title="RAG PDF Chat", theme=gr.themes.Soft())

with demo:
    gr.Markdown("# 📚 RAG PDF Chat Assistant")
    gr.Markdown("Upload a PDF, ask questions, and get AI-powered answers!")
    
    with gr.Row():
        with gr.Column(scale=2):
            model_selector = gr.Dropdown(
                choices=["tiiuae/falcon-rw-1b", "meta-llama/llama-2-7b-chat-hf"],
                value="tiiuae/falcon-rw-1b",
                label="🤖 Select Language Model",
                info="Choose the AI model to use for processing"
            )
        with gr.Column(scale=1):
            load_button = gr.Button("🔄 Load Model", variant="primary")
            
    model_status = gr.Textbox(
        label="Model Status", 
        value="✅ Default model loaded",
        interactive=False
    )
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(
                label="📄 Upload PDF", 
                file_types=[".pdf"],
                file_count="single"
            )
            
        with gr.Column():
            summary_output = gr.Textbox(
                label="📋 Document Summary & Status", 
                lines=8,
                placeholder="Upload a PDF to see its summary here..."
            )
    
    gr.Markdown("---")
    
    with gr.Row():
        user_input = gr.Textbox(
            label="❓ Ask a Question", 
            placeholder="Type your question about the PDF here...",
            lines=2
        )
        
    with gr.Row():
        submit_button = gr.Button("🚀 Ask Question", variant="primary")
        clear_button = gr.Button("🗑️ Clear Chat", variant="secondary")
    
    chat_output = gr.Textbox(
        label="🤖 AI Response", 
        lines=6,
        placeholder="AI responses will appear here..."
    )
    
    # Event handlers
    load_button.click(
        fn=update_llm, 
        inputs=[model_selector], 
        outputs=[model_status]
    )
    
    pdf_input.change(
        fn=process_pdf, 
        inputs=[pdf_input], 
        outputs=[summary_output]
    )
    
    submit_button.click(
        fn=chat_with_pdf, 
        inputs=[user_input], 
        outputs=[chat_output]
    )
    
    user_input.submit(
        fn=chat_with_pdf, 
        inputs=[user_input], 
        outputs=[chat_output]
    )
    
    clear_button.click(
        fn=lambda: ("", ""),
        outputs=[user_input, chat_output]
    )