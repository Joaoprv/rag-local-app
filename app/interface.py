import gradio as gr
from app.pdf_loader import extract_text_from_pdf
from app.summarizer import summarize
from app.chat_rag import build_vector_store, get_chat_chain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


def load_llm():
    model_id = "tiiuae/falcon-rw-1b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()
chat_chain = None

def process_pdf(file):
    global chat_chain
    text = extract_text_from_pdf(file.name)
    summary = summarize(text, llm)
    vectorstore = build_vector_store(text)
    chat_chain = get_chat_chain(llm, vectorstore)
    return summary

def chat_with_pdf(user_input):
    if chat_chain:
        response = chat_chain.run(user_input)
        return response
    return "PDF ainda não foi carregado."

with gr.Blocks() as demo:
    with gr.Row():
        pdf_input = gr.File(label="Faça upload de um PDF")
        summary_output = gr.Textbox(label="Sumário", lines=10)
    with gr.Row():
        user_input = gr.Textbox(label="Sua pergunta")
        chat_output = gr.Textbox(label="Resposta")

    pdf_input.change(process_pdf, inputs=pdf_input, outputs=summary_output)
    user_input.submit(chat_with_pdf, inputs=user_input, outputs=chat_output)

demo.launch()