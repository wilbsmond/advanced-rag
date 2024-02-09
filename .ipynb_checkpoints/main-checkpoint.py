import os
from dotenv import load_dotenv
import openai
from llama_index import download_loader, SimpleDirectoryReader, Document
from llama_index.llms import OpenAI
from trulens_eval import Tru

from sentence_window_retrieval import build_sentence_window_index, get_sentence_window_query_engine
from auto_merging_retrieval import build_automerging_index, get_automerging_query_engine
from trulens_utils import trulens_recorder, run_evals

_ = load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_document():
    documents = SimpleDirectoryReader(
        input_files=["./docs/eBook-How-to-Build-a-Career-in-AI.pdf"]
    ).load_data()

    document = Document(text="\n\n".join([doc.text for doc in documents]))

    return documents, document

def get_eval_questions(path_eval_questions):
    eval_questions = []
    with open(path_eval_questions, 'r') as file:
        for line in file:
            # Remove newline character and convert to integer
            item = line.strip()
            eval_questions.append(item)
    return eval_questions

if __name__ == "__main__":
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

    # Load documents and eval questions
    documents, document = load_document()
    eval_questions = get_eval_questions("./eval_questions/generated_questions_01_05.text")
    Tru().reset_database()

    # Sentence window retrieval
    
    sentence_window_size = 1
    sentence_index = build_sentence_window_index(
        [document],
        llm=llm,
        embed_model="local:BAAI/bge-small-en-v1.5",
        sentence_window_size=sentence_window_size,
        save_dir="./sentence_index",
    )
    sentence_window_engine = get_sentence_window_query_engine(sentence_index, similarity_top_k=6)

    # Evaluate with Trulens
    tru_recorder = trulens_recorder(
        sentence_window_engine,
        app_id=f'sentence window engine {sentence_window_size}'
    )
    run_evals(eval_questions, tru_recorder, sentence_window_engine)
    Tru().get_leaderboard(app_ids=[])
    Tru().run_dashboard()
    """
    # Auto merging retrieval
    auto_merging_index_0 = build_automerging_index(
        documents,
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
        embed_model="local:BAAI/bge-small-en-v1.5",
        save_dir="merging_index_0",
        chunk_sizes=[2048,512],
    )
    auto_merging_engine_0 = get_automerging_query_engine(
        auto_merging_index_0,
        similarity_top_k=12,
        rerank_top_n=6,
    )
    tru_recorder = trulens_recorder(
        auto_merging_engine_0,
        app_id ='app_0'
    )
    run_evals(eval_questions, tru_recorder, auto_merging_engine_0)
    Tru().get_leaderboard(app_ids=[])
    Tru().run_dashboard()
    """