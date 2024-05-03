import random
from typing import List, Tuple, Union, Dict
from inference import askLLM
import time
import os
from PIL import Image
from BucketLink import get_s3_file_url

#model configurations
from config import config
from models import ChatModel
from role_prompt import role_prompt

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage

from dotenv import load_dotenv
load_dotenv()

INIT_MESSAGE = {
    "role": "assistant",
    "content": "Hi! I'm your DorNea, you're financial assistant. How may I help you?",
}

CLAUDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="input"),
    ]
)

class StreamHandler(BaseCallbackHandler):
    """
    Callback handler to stream the generated text to Streamlit.
    """

    def __init__(self, container: st.container) -> None:
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        Append the new token to the text and update the Streamlit container.
        """
        self.text += token
        self.container.markdown(self.text)


def set_page_config() -> None:
    """
    Set the Streamlit page configuration.
    """
    st.set_page_config(page_title="🌍 Chat with DorNea", layout="wide")
    st.title("🌍 Chat with DorNea")


def render_sidebar() -> Tuple[Dict, int, str]:
    """
    Render the sidebar UI and return the inference parameters.
    """
    with st.sidebar:
        # st.markdown("## Inference Parameters")
        model_name_select = st.selectbox(
            'Model',
            list(config["models"].keys()),
            key=f"{st.session_state['widget_key']}_Model_Id",
        )
        
        role_select = st.selectbox(
            'Role',
            ["Default", "Translator", "Writer", "Custom"],
            key=f"{st.session_state['widget_key']}_role_Id",
        )
        # Set the initial value of the text area based on the selected role
        role_prompt_text = role_prompt.get(role_select, "")

        st.session_state["model_name"] = model_name_select

        model_config = config["models"][model_name_select]

        system_prompt_disabled = model_config.get("system_prompt_disabled", False)
        system_prompt = st.text_area(
            "System Prompt",
            # value=model_config.get("default_system_prompt", ""),
            value = role_prompt_text,
            key=f"{st.session_state['widget_key']}_System_Prompt",
            disabled=system_prompt_disabled,
        )
        with st.container():
            col1, col2 = st.columns(2)
            with col1:   
                web_local = st.selectbox(
                    'Web or Local',
                    ('Local', 'Web'),
                    key=f"{st.session_state['widget_key']}_Web",
                )     
            with col2:  
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=model_config.get("temperature", 1.0),
                    step=0.1,
                    key=f"{st.session_state['widget_key']}_Temperature",
                )
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                top_p = st.slider(
                    "Top-P",
                    min_value=0.0,
                    max_value=1.0,
                    value=model_config.get("top_p", 1.0),
                    step=0.01,
                    key=f"{st.session_state['widget_key']}_Top-P",
                )
            with col2:
                top_k = st.slider(
                    "Top-K",
                    min_value=1,
                    max_value=model_config.get("max_top_k", 500),
                    value=model_config.get("top_k", 500),
                    step=5,
                    key=f"{st.session_state['widget_key']}_Top-K",
                )
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                max_tokens = st.slider(
                    "Max Token",
                    min_value=0,
                    max_value=4096,
                    value=model_config.get("max_tokens", 4096),
                    step=8,
                    key=f"{st.session_state['widget_key']}_Max_Token",
                )
            with col2:
                memory_window = st.slider(
                    "Memory Window",
                    min_value=0,
                    max_value=10,
                    value=model_config.get("memory_window", 10),
                    step=1,
                    key=f"{st.session_state['widget_key']}_Memory_Window",
                )

    model_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
    }
    if not model_config.get("system_prompt_disabled", False):
        model_kwargs["system"] = system_prompt

    return model_kwargs, memory_window, web_local


def init_conversationchain(chat_model: ChatModel, memory_window: int) -> ConversationChain:
    """
    Initialize the ConversationChain with the given parameters.
    """
    conversation = ConversationChain(
        llm=chat_model.llm,
        verbose=True,
        memory=ConversationBufferWindowMemory(
            k=memory_window,
            ai_prefix="Assistant",
            chat_memory=StreamlitChatMessageHistory(),
            return_messages=True,
        ),
        prompt=CLAUDE_PROMPT,
    )

    # Store LLM generated responses
    if "messages" not in st.session_state:
        st.session_state.messages = [INIT_MESSAGE]

    return conversation


def generate_response(
    conversation: ConversationChain, input: Union[str, List[dict]]
) -> str:
    """
    Generate a response from the conversation chain with the given input.
    """
    return conversation.invoke(
        {"input": input}, {"callbacks": [StreamHandler(st.empty())]}
    )


def new_chat() -> None:
    """
    Reset the chat session and initialize a new conversation chain.
    """
    st.session_state["messages"] = [INIT_MESSAGE]
    st.session_state["langchain_messages"] = []
    st.session_state["file_uploader_key"] = random.randint(1, 100)


def display_chat_messages(
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]
) -> None:
    """
    Display chat messages and uploaded images in the Streamlit app.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            
            """
            if uploaded_files and "images" in message and message["images"]:
                display_images(message["images"], uploaded_files)
            """
    
            if message["role"] == "user":
                display_user_message(message["content"])

            if message["role"] == "assistant":
                display_assistant_message(message["content"])


def display_images( #to be implemented
    image_ids: List[str],
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile],
) -> None:
    """
    Display uploaded images in the chat message.
    """
    num_cols = 10
    cols = st.columns(num_cols)
    i = 0

    for image_id in image_ids:
        for uploaded_file in uploaded_files:
            if image_id == uploaded_file.file_id:
                if uploaded_file.type.startswith('image/'):
                    img = Image.open(uploaded_file)

                    with cols[i]:
                        st.image(img, caption="", width=75)
                        i += 1

                    if i >= num_cols:
                        i = 0
                elif uploaded_file.type in ['text/plain', 'text/csv', 'text/x-python-script']:
                    if uploaded_file.type == 'text/x-python-script':
                        st.write(f"🐍 Uploaded Python file: {uploaded_file.name}")
                    else:
                        st.write(f"📄 Uploaded text file: {uploaded_file.name}")
                elif uploaded_file.type == 'application/pdf':
                    st.write(f"📑 Uploaded PDF file: {uploaded_file.name}")
                    

def display_user_message(message_content: Union[str, List[dict]]) -> None:
    """
    Display user message in the chat message.
    """
    if isinstance(message_content, str):
        message_text = message_content
    elif isinstance(message_content, dict):
        message_text = message_content["input"][0]["content"][0]["text"]
    else:
        message_text = message_content[0]["text"]

    message_content_markdown = message_text.split('</context>\n\n', 1)[-1]
    st.markdown(message_content_markdown)


def display_assistant_message(message_content: Union[str, dict]) -> None:
    """
    Display assistant message in the chat message.
    """
    if isinstance(message_content, str):
        st.markdown(message_content)
    elif "response" in message_content:
        st.markdown(message_content["response"])


def langchain_messages_format(
    messages: List[Union["AIMessage", "HumanMessage"]]
) -> List[Union["AIMessage", "HumanMessage"]]:
    """
    Format the messages for the LangChain conversation chain.
    """

    for i, message in enumerate(messages):
        if isinstance(message.content, list):
            if "role" in message.content[0]:
                if message.type == "ai":
                    message = AIMessage(message.content[0]["content"])
                if message.type == "human":
                    message = HumanMessage(message.content[0]["content"])
                messages[i] = message
    return messages

 
'''def web_or_local(prompt: str, web_local: str) -> str:
    if web_local == "Web":
        search = SerpAPIWrapper()
        search_text = search.run(prompt)
        web_contect = "Here is the web search result: \n\n<search>\n\n" + str(search_text) + "\n\n</search>\n\n"
        prompt = web_contect + prompt
        return prompt
    else:
        return prompt'''
    
def main() -> None:
    """
    Main function to run the Streamlit app.
    """
    set_page_config()  # Set Streamlit page configuration

    # Generate a unique widget key only once
    if "widget_key" not in st.session_state:  # Check if widget_key is not in session state
        st.session_state["widget_key"] = str(random.randint(1, 1000000))  # Generate and assign a random key

    # Add a button to start a new chat
    st.sidebar.button("New Chat", on_click=new_chat, type="primary") 

    model_kwargs, memory_window, web_local = render_sidebar()  
    chat_model = ChatModel(st.session_state["model_name"], model_kwargs) 
    conv_chain = init_conversationchain(chat_model, memory_window)  # TODO: Initialize conversation chain

    # Image uploader
    if "file_uploader_key" not in st.session_state:  
        st.session_state["file_uploader_key"] = 0 

    #model_config = config["models"][st.session_state["model_name"]]  # Get model configuration from config file

    '''# TODO: Get images from previous messages
    message_images_list = [
        image_id
        for message in st.session_state.messages  # Loop through previous messages in session state
        if message["role"] == "user"  # Check if message role is 'user'
        and "images" in message  # Check if 'images' key is in message
        and message["images"]  # Check if 'images' key has content
        for image_id in message["images"]  # Extract image IDs from 'images' key
    ]'''

    prompt = st.chat_input()  # Display chat input widget and get user input
    
    if prompt:  # Check if user provided a prompt
        formatted_prompt = chat_model.format_prompt(prompt)  # Format the prompt for the chat model
        st.session_state.messages.append({"role": "user", "content": formatted_prompt}) 
        with st.chat_message("user", avatar="nordea.png"):  # Display user's prompt with specified avatar
            st.markdown(prompt)

    # Modify langchain_messages format
    st.session_state["langchain_messages"] = langchain_messages_format(
        st.session_state["langchain_messages"]  # Format language chain messages
    )

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant": 
        with st.chat_message("assistant", avatar="🤖"):  

            response, sources = askLLM(prompt)  # Generate response using language model and get sources
            sources = set(sources)

            def stream_data(delay = 0.02):  # Define generator function for streaming data with optional delay
                for word in response.split(" "):  # Split response into words
                    yield word + " "  # Yield each word with a space
                    time.sleep(delay)  # Introduce delay between yielding words      

            response = st.write_stream(stream_data)
            st.write("Sources:") 
            for s in sources: 
                filename = os.path.basename(s)  
                url = get_s3_file_url(bucket_name, filename)  # Get URL of the file from S3 bucket
                # Format the filename as clickable text to redirect to the URL
                clickable_text = f"[{filename}]({url})"

                st.write("- ", clickable_text)
                
        st.session_state.messages.append({"role": "assistant", "content": response}) 


if __name__ == "__main__":
    main()  
