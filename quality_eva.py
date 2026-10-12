from query_bot import create_engine
from llama_index.core import PromptTemplate,QueryBundle
from setting import setting
import re
from tqdm import tqdm

setting=setting()
local_retrieves_path=[setting.test_retrieves_to_local_path_Nn,setting.test_retrieves_to_local_path_FNn,setting.test_retrieves_to_local_path_Nr]  #提前准备的本地retrieves地址
que_path=setting.que_path
ans_path=setting.ans_path
json_path=setting.json_path
test_len=setting.rag_test_len


query_engine=create_engine()


#直接获取检索文档
def get_retrieves_list():
    import json
    print('正在检索中...')
    retrieves = []
    with open(json_path, encoding='utf-8') as f:
        rows = json.load(f)[:test_len]

    pbar = tqdm(total=len(rows), desc="处理中")
    for r in rows:
        retrieve_doc=query_engine.retrieve(QueryBundle(r['query']))
        retrieves.append(retrieve_doc)
        pbar.update(1)
    print("检索完成\n")
    return rows,retrieves


#使用提前检索好的结果
def get_retrieves_list_from_local(retrieve_path):
    import pickle
    import json
    print(f"正在使用读取{retrieve_path['name']} retrievers数据...")
    with open(json_path, encoding='utf-8') as f:
        rows = json.load(f)
    with open(retrieve_path['path'], 'rb') as f:
        retrievers = pickle.load(f)
    print("读取成功")
    return rows,retrievers



#用于评估的三个函数
def is_hit(r, name):
    return name == r['name'] or (r.get('alias') and name in r['alias'])
def recall_at_k(k,rows,retrieves):
    hit = 0
    for i,r in enumerate(rows):
        docs = retrieves[i][:k]
        names = [d.node.metadata.get('病名') for d in docs]
        hit += any(is_hit(r, n) for n in names)
    return hit / len(rows)
def mrr(rows,retrieves):
    total = 0.0
    for i,r in enumerate(rows):
        for rank, d in enumerate(retrieves[i], 1):
            if is_hit(r, d.node.metadata.get('病名')):
                total += 1 / rank
                break
    return total / len(rows)



def rag_eva_main():
    rows,retrieves = get_retrieves_list()
    print('rag质量评估中...')
    for k in (1, 3, 5, 10):
        print(f'Recall前{k} = {recall_at_k(k,rows,retrieves):.3f}')
    print(f'MRR = {mrr(rows,retrieves):.3f}')


def rag_eva_local_main():

    for retrieve_path in local_retrieves_path:
        rows,retrieves = get_retrieves_list_from_local(retrieve_path)
        print('rag质量评估中...')
        print('-'*10+f"{retrieve_path['name']}"+'-'*10)
        for k in (1, 3, 5, 10):
            print(f'Recall前{k} = {recall_at_k(k,rows,retrieves):.3f}')
        print(f'MRR = {mrr(rows,retrieves):.3f}')
        print('-'*25,end='\n\n')

if __name__ == '__main__':
    rag_eva_local_main() #用我提前检索好的数据测试
    #rag_eva_main()      #检索并进行测试，在此之前设置好test_len：测试数据长度