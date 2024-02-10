import os
from dotenv import load_dotenv
import openai
from llama_index import SimpleDirectoryReader, Document#, download_loader
from llama_index import ServiceContext, VectorStoreIndex#, StorageContext, load_index_from_storage
from llama_index.llms import OpenAI#, MistralAI
#from llama_index.embeddings import MistralAIEmbedding
#from trulens_eval import Tru
import streamlit as st

import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
notion_token = os.getenv('NOTION_INTEGRATION_TOKEN')

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(
            input_files=["./db_docs/docs/eBook-How-to-Build-a-Career-in-AI.pdf"]
        )
        docs = reader.load_data()

        llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
        embed_model = "local:BAAI/bge-small-en-v1.5"
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)#, system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

if __name__ == "__main__":


    st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
    st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")
    st.info("Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")
            
    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
        ]

    index = load_data()

    if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
            st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) # Add response to message history