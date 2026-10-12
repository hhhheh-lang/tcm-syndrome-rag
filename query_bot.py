from llama_index.core import Settings,load_index_from_storage,StorageContext,get_response_synthesizer, QueryBundle,VectorStoreIndex,PromptTemplate
from llama_index.llms.dashscope import DashScope,DashScopeGenerationModels
from llama_index.embeddings.dashscope import DashScopeEmbedding,DashScopeTextEmbeddingModels
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.postprocessor import SimilarityPostprocessor,LLMRerank
from chromadb.utils import embedding_functions
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import os
from setting import setting


setting=setting()
Settings.llm=setting.llm
Settings.embed_model=setting.embed_model


doc_path=setting.doc_path
chroma_storage_path=setting.chroma_storage_path
collection_name=setting.collection_name


def init_chroma():
    chroma_client = chromadb.PersistentClient(path=chroma_storage_path)
    try:
        chroma_collection=chroma_client.get_collection(collection_name)
    except Exception as e:
        raise ValueError("-"*10+f'!!!chroma集合:{collection_name} 未创建'+"-"*10,end='\n\n')
    return chroma_collection



def create_index():
    chroma_collection=init_chroma()
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )
    return index





def init_prompt():
    qa_prompt_tmpl_str = (
        "上下文为中医各证候资料，包含证候名称与对应临床表现。\n"
        "------------------------\n"
        "{context_str}\n"
        "------------------------\n"
        "任务：根据用户描述的症状，从上面上下文资料中匹配对应的中医证候。\n"
        "作为一个医疗人工智能助手，你的回答要尽可能严谨。\n"
        "仅使用上下文提供的信息，禁止使用模型内置知识库。\n"
        "请输出匹配的证候名称，并列出匹配依据（对应的症状），并在最后给出建议。\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

    refine_prompt_tmpl_str = (
        "原始查询如下：{query_str}"
        "我们提供了现有答案：{existing_answer}"
        "我们有机会通过下面的更多上下文来完善现有答案（仅在需要时）。"
        "------------"
        "{context_msg}"
        "------------"
        "考虑到新的上下文，优化原始答案以更好地回答查询。 如果上下文没有用，请返回原始答案。"
        "Refined Answer:"
    )
    refine_prompt_tmpl = PromptTemplate(refine_prompt_tmpl_str)
    return qa_prompt_tmpl, refine_prompt_tmpl


def create_engine():
    index = create_index()
    retriever = index.as_retriever(similarity_top_k=10)

    fusion_retriever = QueryFusionRetriever(
        [index.as_retriever()],
        similarity_top_k=10,
        num_queries=3,
        use_async=False,
    )

    sp = SimilarityPostprocessor(similarity_cutoff=0.6)
    rerank = LLMRerank(top_n=5)
    query_engine = RetrieverQueryEngine.from_args(
        fusion_retriever,
        node_postprocessors=[],
        response_synthesizer=get_response_synthesizer(
            response_mode=ResponseMode.SIMPLE_SUMMARIZE,
        ),
    )

    qa_prompt_tmpl, refine_prompt_tmpl = init_prompt()
    query_engine.update_prompts(
        {
            "response_synthesizer:text_qa_template": qa_prompt_tmpl,
        }
    )
    return query_engine


def print_engine_prompt(query_engine):
    prompts_dict = query_engine.get_prompts()
    for key, value in prompts_dict.items():
        print()
        print(key, value)
        print()



def print_retrieves(query_engine,query):
    rte_docs = query_engine.retrieve(QueryBundle(query))
    print("-" * 10 + f"召回{len(rte_docs)}个片段" + "-" * 10)
    for i, rte_doc in enumerate(rte_docs, start=1):
        print('-' * 10 + f'doc{i}' + '-' * 10)
        print('编号：' + rte_doc.metadata['编号'])
        print('病名：' + rte_doc.metadata['病名'])
        print(rte_doc.text)
        print('-' * 10 + f'END' + '-' * 10)



def engine_query_main():
    query_engine = create_engine()
    while True:
        query = input('输入症状：')              #"小便清长，畏寒、肢冷，可能是哪些症候"
        print_retrieves(query_engine, query)   #追踪哪些文档片段被检索

        print("\n思考中...")
        respond = query_engine.query(query)
        print(f'Assistant回复：\n{respond}')
        print()


if __name__ == '__main__':
    engine_query_main()





