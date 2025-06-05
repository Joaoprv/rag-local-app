from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def get_summarizer_chain(llm):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Resuma o seguinte texto em português:\n\n{text}"
    )
    return LLMChain(llm=llm, prompt=prompt)

def summarize(text, llm):
    chain = get_summarizer_chain(llm)
    return chain.invoke(text)