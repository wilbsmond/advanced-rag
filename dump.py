"""
def build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index",
    chunk_sizes=None,
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context, service_context=merging_context
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=merging_context,
        )
    return automerging_index
"""
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
