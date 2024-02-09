"""
def build_query_engine(llm, embed_model, mode, documents):
    if mode == "basic":
        query_engine = index.as_query_engine(similarity_top_k=2)

    if mode == "sentence_window":
        sentence_window_size = 1

        document = join_documents(documents)
        sentence_index = build_sentence_window_index(
            [document],
            llm=llm,
            embed_model=embed_model,
            sentence_window_size=sentence_window_size,
            save_dir="./db_index/sentence_index",
        )
        query_engine = get_sentence_window_query_engine(sentence_index, similarity_top_k=6)

    elif mode == "auto_merging":
        chunk_sizes = [2048,512]

        auto_merging_index = build_automerging_index(
            documents,
            llm=llm,
            embed_model=embed_model,
            save_dir="./db_index/merging_index",
            chunk_sizes=chunk_sizes,
        )
        query_engine = get_automerging_query_engine(
            auto_merging_index,
            similarity_top_k=12,
            rerank_top_n=6,
        )
    return query_engine
"""
