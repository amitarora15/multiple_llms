import os
import logging
import tempfile
from dotenv import load_dotenv

# Disable tqdm completely by setting environment variable
os.environ["TQDM_DISABLE"] = "true"

import openai
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

import torch  # Ensure torch is imported for device checks
from transformers import AutoModelForCausalLM, AutoTokenizer
import whisper  # For audio transcription

# Step 1: Load Required Packages and API Keys
# ---------------------------------------------------

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_inference.log"),
        logging.StreamHandler()
    ]
)

try:
    # API keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')  # Optional

    if not OPENAI_API_KEY:
        logging.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        raise ValueError("OpenAI API key not found.")

    # Set OpenAI API key
    openai.api_key = OPENAI_API_KEY

    # Set HuggingFace API key if needed
    if HUGGINGFACE_API_KEY:
        os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY
        logging.info("HuggingFace API key set.")

    logging.info("Successfully loaded API keys.")

except Exception as e:
    logging.error(f"Error loading API keys: {e}")
    raise

# Step 2: Instantiate Different LLMs
# ---------------------------------------------------

class LLMManager:
    def __init__(self, openai_api_key, huggingface_api_key=None):
        self.openai_api_key = openai_api_key
        self.huggingface_api_key = huggingface_api_key
        self.llms = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.instantiate_llms()

    def instantiate_llms(self):
        try:
            # OpenAI GPT-3.5-turbo
            self.llms['gpt-3.5-turbo'] = 'gpt-3.5-turbo'
            logging.info("OpenAI GPT-3.5-turbo instantiated.")

            # OpenAI GPT-4
            self.llms['gpt-4'] = 'gpt-4'
            logging.info("OpenAI GPT-4 instantiated.")

            # HuggingFace Zephyr-7B Beta
            model_name = "HuggingFaceH4/zephyr-7b-beta"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map='auto',
                torch_dtype=torch.float16,
                load_in_8bit=False,
                pad_token_id=tokenizer.eos_token_id
            )
            self.llms['zephyr-7b-beta'] = {
                'model': model,
                'tokenizer': tokenizer
            }
            logging.info("HuggingFace Zephyr-7B Beta instantiated.")

        except Exception as e:
            logging.error(f"Error instantiating LLMs: {e}")
            raise

    def get_llm_names(self):
        return list(self.llms.keys())

# Initialize LLMManager
llm_manager = LLMManager(OPENAI_API_KEY, HUGGINGFACE_API_KEY)

# Step 3: Create a Function to Generate Responses
# ---------------------------------------------------

class ResponseGenerator:
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager

    def generate_response(self, user_input, selected_llm=None, conversation_history=None):
        responses = {}
        try:
            if selected_llm:
                llm_names = [selected_llm]
                logging.info(f"Generating response using selected LLM: {selected_llm}")
            else:
                llm_names = self.llm_manager.get_llm_names()
                logging.info("Generating responses using all available LLMs.")

            for llm in llm_names:
                if llm in ['gpt-3.5-turbo', 'gpt-4']:
                    # Prepare messages with conversation history
                    logging.info(f"Inside Response Generation {llm}.")
                    messages = conversation_history.copy() if conversation_history else []
                    messages.append({"role": "user", "content": user_input})

                    response = openai.ChatCompletion.create(
                        model=llm,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=150
                    )
                    responses[llm] = response['choices'][0]['message']['content'].strip()
                    logging.info(f"Response generated from {llm}.")

                elif llm == 'zephyr-7b-beta':
                    logging.info(f"Inside Response Generation {llm}.")
                    tokenizer = self.llm_manager.llms[llm]['tokenizer']
                    model = self.llm_manager.llms[llm]['model']
                    logging.info(f"After tokenizer & model {llm}.") 
                    # Prepare input prompt with conversation history
                    prompt = ""
                    if conversation_history:
                        for msg in conversation_history:
                            if msg['role'] == 'user':
                                prompt += f"User: {msg['content']}\n"
                            elif msg['role'] == 'assistant':
                                prompt += f"Assistant: {msg['content']}\n"
                    prompt += f"User: {user_input}\nAssistant:"
                    logging.info(f"Prompt for {llm} is {prompt}.")
                    logging.info(f"Device is {self.llm_manager.device}")
                    inputs = tokenizer(prompt, return_tensors="pt").to(self.llm_manager.device)
                    logging.info(f"Inputs collected for {llm}")
                    outputs = model.generate(
                        **inputs,
                        max_length=50,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    logging.info(f"response collected for {llm} is {response}")
                    # Extract the assistant's reply
                    response = response.split("Assistant:")[-1].strip()
                    responses[llm] = response
                    logging.info(f"Response generated from {llm}.")

                else:
                    logging.warning(f"LLM {llm} is not recognized.")
                    responses[llm] = "LLM not supported."

        except Exception as e:
            logging.error(f"Error generating responses: {e}")
            responses['error'] = str(e)

        return responses

# Initialize ResponseGenerator
response_generator = ResponseGenerator(llm_manager)

# Step 4: Include Conversation Memory
# ---------------------------------------------------

class ChatMemory:
    def __init__(self):
        self.history = []

    def add_user_message(self, message):
        self.history.append({"role": "user", "content": message})
        logging.info("User message added to history.")

    def add_assistant_response(self, llm, message):
        self.history.append({"role": "assistant", "content": f"({llm}) {message}"})
        logging.info(f"Assistant response from {llm} added to history.")

    def get_history(self):
        # Convert history to the format required by OpenAI
        formatted_history = []
        for msg in self.history:
            if msg['role'] == 'assistant':
                # Extract the actual message without the LLM identifier
                content = msg['content']
                # Assuming format "(llm_name) response"
                if ') ' in content:
                    content = content.split(') ', 1)[1]
                formatted_history.append({"role": "assistant", "content": content})
            else:
                formatted_history.append(msg)
        return formatted_history

    def clear_history(self):
        self.history = []
        logging.info("Conversation history cleared.")

# Initialize ChatMemory
chat_memory = ChatMemory()

# Step 5: Add Audio Processing
# ---------------------------------------------------

class AudioProcessor:
    def __init__(self):
        try:
            self.model = whisper.load_model("base")  # Choose model size as needed
            logging.info("Whisper model loaded for audio processing.")
        except Exception as e:
            logging.error(f"Error loading Whisper model: {e}")
            raise

    def transcribe_audio(self, audio_path):
        try:
            result = self.model.transcribe(audio_path)
            transcription = result["text"].strip()
            logging.info(f"Audio transcribed to text: {transcription}")
            return transcription
        except Exception as e:
            logging.error(f"Error transcribing audio: {e}")
            return "Error transcribing audio."

# Initialize AudioProcessor
audio_processor = AudioProcessor()

# Step 6: Create a User Interface with Gradio
# ---------------------------------------------------

def submit_message(user_message, selected_llm):
    if not user_message.strip():
        return chat_memory.history, "Please enter a message."

    chat_memory.add_user_message(user_message)
    logging.info(f"User submitted message: {user_message} with selected LLM: {selected_llm}")

    if selected_llm == "All":
        selected = None
    else:
        selected = selected_llm

    responses = response_generator.generate_response(
        user_input=user_message,
        selected_llm=selected,
        conversation_history=chat_memory.get_history()
    )

    # Append responses to chat history
    for llm, resp in responses.items():
        if llm != 'error':
            chat_memory.add_assistant_response(llm, resp)
        else:
            chat_memory.add_assistant_response("System", resp)

    # Prepare messages for display
    display_messages = []
    for msg in chat_memory.history:
        role = msg['role']
        content = msg['content']
        if role == 'user':
            display_messages.append((f"User", content))
        elif role == 'assistant':
            # Extract LLM name if present
            if msg['content'].startswith('(System)'):
                display_messages.append(("System", content.replace("(System) ", "")))
            elif msg['content'].startswith('('):
                llm_end = msg['content'].find(')')
                llm_name = msg['content'][1:llm_end]
                actual_content = msg['content'][llm_end+2:]
                display_messages.append((f"Response From {llm_name}", actual_content))
            else:
                display_messages.append(("Assistant", content))

    return display_messages, ""

def submit_text_message(user_message, selected_llm):
    return submit_message(user_message, selected_llm)

def submit_audio_message(audio_file, selected_llm):
    if audio_file is None:
        return chat_memory.history, "Please provide an audio file."

    # Transcribe the audio to text
    user_message = audio_processor.transcribe_audio(audio_file)
    if not user_message:
        return chat_memory.history, "Could not transcribe audio."

    return submit_message(user_message, selected_llm)

def clear_chat():
    chat_memory.clear_history()
    return [], "Chat history cleared."

# Data Visualization Function (Optional)
def visualize_history(history):
    try:
        # Count the number of responses from each LLM
        llm_counts = {}
        for msg in history:
            if msg['role'] == 'assistant':
                if msg['content'].startswith('('):
                    llm_end = msg['content'].find(')')
                    llm_name = msg['content'][1:llm_end]
                    llm_counts[llm_name] = llm_counts.get(llm_name, 0) + 1

        if llm_counts:
            df = pd.DataFrame(list(llm_counts.items()), columns=['LLM', 'Responses'])
            plt.figure(figsize=(8, 6))
            plt.bar(df['LLM'], df['Responses'], color='skyblue')
            plt.xlabel('LLM')
            plt.ylabel('Number of Responses')
            plt.title('LLM Response Count')
            plt.tight_layout()
            # Save the plot to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                plt.savefig(tmp_img.name)
                plt.close()
                logging.info("Data visualization created successfully.")
                return tmp_img.name
        else:
            return None

    except Exception as e:
        logging.error(f"Error in data visualization: {e}")
        return None

# Define the UI layout with audio components
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Multiple LLMs Inference System")

    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(
                label="Enter your message:",
                placeholder="Type your message here...",
                lines=2
            )
            llm_selection = gr.Dropdown(
                choices=["All"] + llm_manager.get_llm_names(),
                value="All",
                label="Select LLM"
            )
            submit_text_btn = gr.Button("Submit Text")
            gr.Markdown("----")
            audio_input = gr.Audio(source="microphone", type="filepath", label="Or Record Your Message:")
            submit_audio_btn = gr.Button("Submit Audio")
            clear_btn = gr.Button("Clear Chat")
            visualize_btn = gr.Button("Visualize LLM Usage")
        with gr.Column():
            chat_box = gr.Chatbot(label="Conversation")
            status_text = gr.Textbox(label="Status", interactive=False, lines=1)
            visualization = gr.Image(label="LLM Usage Visualization")

    # Bind the submit and clear functions
    submit_text_btn.click(
        fn=submit_text_message,
        inputs=[user_input, llm_selection],
        outputs=[chat_box, status_text]
    )
    submit_audio_btn.click(
        fn=submit_audio_message,
        inputs=[audio_input, llm_selection],
        outputs=[chat_box, status_text]
    )
    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chat_box, status_text]
    )
    visualize_btn.click(
        fn=lambda: visualize_history(chat_memory.history),
        inputs=[],
        outputs=visualization
    )

    # Optional: Add more UI elements as needed

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()
# multi_llm_inference_fixed.py
