from llama_index.llms.dashscope import DashScope,DashScopeGenerationModels
from llama_index.embeddings.dashscope import DashScopeEmbedding,DashScopeTextEmbeddingModels
from dotenv import load_dotenv
import os
load_dotenv()

class setting:
    def __init__(self):
        self.api_key=os.getenv('DASHSCOPE_API_KEY')
        self.llm = DashScope(model=DashScopeGenerationModels.QWEN_MAX, api_key=self.api_key)
        self.embed_model = DashScopeEmbedding(model=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V1)

        self.doc_path = './data'
        self.file_name = 'demo.txt'
        self.chroma_storage_path = './chroma_storage'
        self.collection_name = 'symptom'

        self.que_path = './test_data/症状文档.txt'
        self.ans_path = './test_data/病名文档.txt'
        self.json_path = './test_data/eval_dataset.json'
        self.rag_test_len=10

        self.test_retrieves_to_local_path_Nn = {'path':'retrieve_docs/nodeN_retrieves.pkl','name':'default'}              #default
        self.test_retrieves_to_local_path_FNn= {'path':'retrieve_docs/fusion_nodeN_retrieves.pkl','name':'fusion_query'}  #使用fusion_query的检索结果
        self.test_retrieves_to_local_path_Nr = {'path':'retrieve_docs/nodeR_retrieves.pkl','name':'rerank'}               #对检索结果进行rerank的检索结果
