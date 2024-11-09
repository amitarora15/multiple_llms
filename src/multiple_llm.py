import openai
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint

import gradio as gr

def read_keys():
    with open("keys/openapi.txt") as openai_file:
        openapi_key = openai_file.read()
        os.environ["OPENAI_API_KEY"] = openapi_key

    with open("keys/hugging_face.txt") as huggging_file:
        hugging_key = huggging_file.read()
        os.environ["HF_TOKEN"] = hugging_key


def load_gpt_model(modal_type):
    memory = ConversationBufferMemory()
    model = ChatOpenAI(temperature=0.9, model=modal_type)
    open_ai_llm = ConversationChain(llm=model, memory = memory)          
    return open_ai_llm

def load_hf_model():
    model = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens = 512,
        top_k = 30,
        temperature = 0.9,
        repetition_penalty = 1.03,
    )   
    memory = ConversationBufferMemory()
    hf_llm = ConversationChain(llm=model, memory = memory)  
    return hf_llm    

def generate_response(message, history, modal_types):
    #print("Model is ", modal_type)
    read_keys();
    response = ""
    for model_type in modal_types:
        if model_type.lower() == "gpt-3.5-turbo".lower():
            llm = load_gpt_model(model_type)
            response +=  "gpt-3.5-turbo response => " + llm.predict(input=message) + "\n"
        elif model_type.lower() == "Hugging_Face_Zephyr".lower():
            hf_llm = load_hf_model()
            #print(hf_llm.invoke(input=message))
            hf_response = hf_llm.invoke(input=message)
            hf_response = hf_response['response']
            response +=  "Hugging Face Zephyr response => " + hf_response  + "\n"
        elif model_type.lower() == "gpt-4o-mini".lower():
            llm = load_gpt_model(model_type)
            response += "gpt-4o-mini response => " + llm.predict(input=message)  + "\n"
    return response

def main():
    
    with gr.Blocks() as demo:
        clear = gr.Button("Clear")  
        chat_interface = gr.ChatInterface(
            fn=generate_response, 
            type="messages", 
            theme="soft",
            title="Multi LLM Group-16 Chat Bot",
            textbox=gr.Textbox(placeholder="Type your prompt and select model", container=False, scale=7, label="prompt_textbox"),
            analytics_enabled=True,
            show_progress="full",
            fill_width=True,
            submit_btn=True,
            stop_btn=True,
            additional_inputs=[
                gr.CheckboxGroup(["gpt-3.5-turbo", "Hugging_Face_Zephyr", "gpt-4o-mini"], label="modal_type", info="Which model to select?"),
            ]
        )
        clear.click(lambda: None, None, chat_interface, queue=False)

    demo.launch()

if __name__ == "__main__":
    main()

    