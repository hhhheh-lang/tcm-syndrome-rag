import json
from setting import setting
from llama_index.core import QueryBundle
from query_bot import create_engine
import pickle
from tqdm import tqdm

setting = setting()
json_path=setting.json_path                                    #json文件地址
persist_to_local_path=setting.test_retrieves_to_local_path_Nr  #retrieves持久化地址
query_engine=create_engine()

def main():
    with open(json_path, 'r', encoding='utf-8') as f:
        rows = json.load(f)

    retrievers=[]

    print('开始检索...')
    pbar = tqdm(total=len(rows), desc="处理中")
    for q in rows:
        retriever=query_engine.retrieve(QueryBundle(q['query']))
        retrievers.append(retriever)
        pbar.update(1)
    pbar.close()

    print('检索完成，本地化中...')
    with open(persist_to_local_path, 'wb') as f:
        pickle.dump(retrievers, f)
    print('本地化完成')


if __name__ == '__main__':
    main()

