from setting import setting
import chromadb
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex,Settings,load_index_from_storage,StorageContext,Document
from llama_index.core.node_parser import SentenceSplitter
from chromadb.utils import embedding_functions
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb.errors import NotFoundError
import re
import os

setting=setting()

def doc_splitter():
    print("-"*10+'正在解析文档内容...'+"-"*10)
    doc_path=setting.doc_path            #文档路径
    file_name=setting.file_name          #文档名

    ID_RE = re.compile(r'^\s*(\d+(?:\.\d+)+)\s*$')       # 按编号取正则
    JUNK_RE = re.compile(r'[\s,\-\[\]()（）]+$')          # 清掉行尾 OCR 噪声：, - [ ]


    with open(os.path.join(doc_path,file_name), encoding='utf-8') as f:
        lines = f.read().splitlines()


    id_positions = [i for i, l in enumerate(lines) if ID_RE.match(l)] #取所有编号的索引位置
    documents = []
    for k, i in enumerate(id_positions):
        end = id_positions[k + 1] if k + 1 < len(id_positions) else len(lines)
        body = [l.strip() for l in lines[i + 1:end] if l.strip()]                                  # 只有病名没有描述，丢弃

        code = ID_RE.match(lines[i]).group(1)                 # 获取编号
        name = JUNK_RE.sub('', body[0])                  # 病名
        rest = body[1:]                                       #
        alias = [JUNK_RE.sub('', l) for l in rest if '。' not in l and len(l) < 15]   #取别名，不带句号且长度短的
        desc = ' '.join(l for l in rest if not ('。' not in l and len(l) < 15)).strip()    #取正文内容

        if '临床' in desc:                                        #以 “临床以” 为界分割病因和症状
            reason, symptom = desc.split('临床', 1)
            reason = reason.strip()
            symptom = '临床' + symptom
        else:
            reason, symptom = '', desc

        # 封装document对象；关键：病名写进正文，embedding 才"看得见"它
        text = (f'症状：{symptom}')
        documents.append(Document(
            text=text,
            metadata={'编号': code, '病名': name, '别名':','.join(alias), '病因':reason},
            # 正文里已经有病名了，metadata 不必再重复参与向量计算
            excluded_embed_metadata_keys=['编号', '病名', '别名', '病因'],
        ))

    print(f'解析出 {len(documents)} 条证候')
    return documents



def init_chroma():
    print("-"*10+'正在初始化chroma数据库...'+"-"*10)
    chroma_storage_path=setting.chroma_storage_path
    collection_name=setting.collection_name

    default_ef = embedding_functions.DefaultEmbeddingFunction()
    chroma_client = chromadb.PersistentClient(path=chroma_storage_path)

    try:
        chroma_client.get_collection(name=collection_name)
        chroma_client.delete_collection(name=collection_name)
    except NotFoundError:
        pass

    chroma_collection=chroma_client.get_or_create_collection(
                    name=collection_name,
                    configuration={
                        "hnsw": {
                            "space": "cosine",
                            "ef_search": 100,
                            "ef_construction": 100,
                            "max_neighbors": 16
                        }
                    }
    )
    print('chroma初始化完成')
    return chroma_collection



def main():
    collection_name = setting.collection_name

    documents = doc_splitter()
    chroma_collection = init_chroma()


    Settings.llm=setting.llm
    Settings.embed_model=setting.embed_model
    #Settings.transformations=[SentenceSplitter(chunk_size=300,chunk_overlap=100)]
    #document=SimpleDirectoryReader(doc_path,required_exts=['.txt']).load_data()


    print("-"*10+f'正在创建集合{collection_name},并存储向量数据...'+"-"*10)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context=StorageContext.from_defaults(vector_store=vector_store)
    index=VectorStoreIndex.from_documents(documents,storage_context=storage_context)

    print("-"*10+f"chroma集合{collection_name}构建成功"+"-"*10,end='\n\n')


main()