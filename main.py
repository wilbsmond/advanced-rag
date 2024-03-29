import os
from dotenv import load_dotenv
import openai
from llama_index import SimpleDirectoryReader, Document#, download_loader
from llama_hub.notion import NotionPageReader
from llama_index.llms import OpenAI#, MistralAI
#from llama_index.embeddings import MistralAIEmbedding
from llama_index import ServiceContext, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.node_parser import SentenceWindowNodeParser
#from trulens_eval import Tru
import streamlit as st

from utils.sentence_window_retrieval import build_sentence_window_context
#from utils.auto_merging_retrieval import build_automerging_index, get_automerging_query_engine
#from utils.trulens_utils import trulens_recorder, run_evals

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
notion_token = os.getenv('NOTION_INTEGRATION_TOKEN')
#openai.api_key = st.secrets.OPENAI_API_KEY

def load_documents():
    """
    reader = SimpleDirectoryReader(input_files=["./db/docs/eBook-How-to-Build-a-Career-in-AI.pdf"])
    documents = reader.load_data()
    """
    #NotionPageReader = download_loader('NotionPageReader')
    page_ids = ["491ea0f6b03147bb8dbc78d5ba6d058d"]
    documents = NotionPageReader(integration_token=notion_token).load_data(
        page_ids=page_ids
    )
    #"""
    return documents

def join_documents(documents):
    document = Document(text="\n\n".join([doc.text for doc in documents]))
    return document

def build_retrieval(llm, retrieval_mode):
    if retrieval_mode == "basic":
        context = ServiceContext.from_defaults(llm=llm)

    elif retrieval_mode == "sentence_window":
        context = build_sentence_window_context(llm, sentence_window_size=3)
        """
        sentence_window_size = 3
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=sentence_window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        context = ServiceContext.from_defaults(
            llm=llm,
            node_parser=node_parser,
        )
        """
    return context

def build_index(documents, retrieval_context, retrieval_mode):
    save_dir = f"./db/index/{retrieval_mode}_index"
    if not os.path.exists(save_dir):
        index = VectorStoreIndex.from_documents(documents, service_context=retrieval_context)
        index.storage_context.persist(persist_dir=save_dir)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=retrieval_context,
        )

    return index

@st.cache_resource(show_spinner=False)
def load_data_to_index(_llm, retrieval_mode):
    with st.spinner(text="Loading and indexing the {} docs – hang tight! This should take a few minutes."):
        documents = load_documents()
        retrieval_context = build_retrieval(_llm, retrieval_mode)
        index = build_index(documents, retrieval_context, retrieval_mode)
        return index

if __name__ == "__main__":
    system_prompt = """
                You are an expert on the Blendle's Employee Handbook and your job is to answer questions relating to its contents. 
                Keep your answers based on facts – do not hallucinate features.
                """
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1, system_prompt=system_prompt)
    #llm = MistralAI(model="mistral-medium", api_key=os.getenv("MISTRAL_API_KEY"))
    #embed_model = "local:BAAI/bge-small-en-v1.5"
    #embed_model = MistralAIEmbedding(model_name="mistral-embed", api_key=os.getenv("MISTRAL_API_KEY"))
    retrieval_mode = "sentence_window"


    ## Streamlit --------------------------------------
    #st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
    st.title("Chat with [Blendle's Employee Handbook](https://yolospace.notion.site/Blendle-s-Employee-Handbook-e31bff7da17346ee99f531087d8b133f)")
    st.info("Powered by Advanced RAG and LlamaIndex 💬🦙")
    #st.info("Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")

    # Initialize message history
    if "messages" not in st.session_state.keys(): # Initialize chat message history
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about Blendle's Employee Handbook!"}
        ]

    # Load and index data
    index = load_data_to_index(llm, retrieval_mode)

    # Create chat engine
    if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
  
    # Prompt for user input and display message history
    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Pass query to chat engine and display response
    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) # Add response to message history