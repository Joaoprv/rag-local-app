import gradio as gr
from app.pdf_service import process_pdf
from models.models import AppState, ModelType
from app.llm_service import LLMService
import logging

logger = logging.getLogger(__name__)

# TODO: Move update_llm to LLMService and chat_with_pdf to chat_rag 

def update_llm(model_name_str):
    """Update the LLM using the LLMService."""
    try:
        # Convert string to ModelType enum
        model_type = ModelType(model_name_str)
        logger.info(f"Updating LLM to: {model_type.value}")
        
        new_llm = llm_service.load_model(model_type)
        if new_llm is not None:
            app_state.llm = new_llm
            app_state.current_model = model_type
            app_state.reset_chat_chain()  # Reset chat chain when model changes
            logger.info(f"Successfully updated LLM to: {model_type.value}")
            return f"✅ Model '{model_type.value}' loaded successfully!"
        else:
            logger.warning(f"Failed to load model: {model_type.value}")
            return f"❌ Failed to load model '{model_type.value}'"
    except ValueError:
        return f"❌ Invalid model name: {model_name_str}"
    except Exception as e:
        logger.error(f"Error updating LLM: {str(e)}")
        return f"❌ Error loading model: {str(e)}"


    
def chat_with_pdf(user_input):
    """Chat with PDF"""
    if not user_input.strip():
        logger.warning("Empty chat input received")
        return "Please enter a question."
    
    if app_state.chat_chain is None:
        logger.warning("Chat attempted without processed PDF")
        return "❌ Please upload and process a PDF first!"
    
    try:
        logger.info(f"Processing chat question: {user_input[:100]}...")
        
        config = {"configurable": {"session_id": "default_session"}}
        response = app_state.chat_chain.invoke({"question": user_input}, config)
        
        if isinstance(response, dict):
            result = response.get("answer", response.get("result", str(response)))
        else:
            result = str(response)
        
        logger.info("Chat response generated successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return f"❌ Error generating response: {str(e)}"


llm_service = LLMService()
app_state = AppState()

logger.info("Initializing with default model...")

app_state.llm = llm_service.load_model(ModelType.QWEN)
app_state.current_model = ModelType.QWEN

demo = gr.Blocks(title="RAG PDF Chat", theme=gr.themes.Soft())

with demo:
    gr.Markdown("# 📚 RAG PDF Chat Assistant")
    gr.Markdown("Upload a PDF, ask questions, and get AI-powered answers!")
    
    with gr.Row():
        with gr.Column(scale=2):
            model_selector = gr.Dropdown(
                choices=[model.value for model in ModelType],
                value=ModelType.QWEN.value,
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
        inputs=[pdf_input, gr.State(value=app_state)], 
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