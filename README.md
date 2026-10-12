# 中医临床智能诊疗助手

基于 RAG（检索增强生成）的中医证候诊断助手：输入病人的症状描述，检索《中医临床诊疗术语·证候》知识库，返回匹配的证候与依据，辅助医生诊断（不做最终确诊）。

## 功能特性

- **症状 → 证候**：输入症状描述，检索并生成匹配的证候名称、匹配依据与建议
- **结构化知识库**：1654 条证候按编号切分入库，向量只嵌入「症状」文本，病名/别名/病因作为元数据保留
- **RAG 质量评估**：内置 Recall@k、MRR 检索指标，预留 RAGAS 生成评估（已注释）
- **可配置**：模型、路径、集合名统一在 `setting.py` 维护

## 技术栈

| 模块 | 选型 |
|---|---|
| 大模型 | 通义千问 Qwen-Max（DashScope） |
| 向量模型 | text-embedding-v1（DashScope） |
| 框架 | LlamaIndex |
| 向量数据库 | ChromaDB |
| 评估指标 | Recall@k / MRR |

## 目录结构

```
项目代码/
├── setting.py                    # 配置中心：模型、路径、集合名、评估参数
├── chroma_storage.py             # 解析 demo.txt 并构建向量库
├── query_bot.py                  # 查询引擎封装 + 交互式查询入口
├── quality_eva.py                # RAG 质量评估（Recall@k / MRR）
├── test_retrieves_to_local.py    # 预先检索并持久化检索结果，供 RAG 质量评估离线复用（配合 rag_eva_local_main 使用）
├── .env                          # 存放 DASHSCOPE_API_KEY（需自行填写）
├── requirements.txt              # 依赖清单
├── chroma_storage/               # 向量库持久化数据（运行后生成）
├── test_data/                    # 评估数据集（症状/病名/可溯源 JSON）
└── data/                         # 数据文档
```

## 环境要求

- Python 3.9+
- 通义千问（DashScope）API Key

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key：编辑 .env，填入你的 key；你可以从https://bailian.console.aliyun.com获取你的key。
#    DASHSCOPE_API_KEY=sk-xxxx

# 3. 构建向量库（会删除旧集合并重建）
python chroma_storage.py

# 4. 交互式查询
python query_bot.py
```

## 使用说明

### 交互式查询

```bash
python query_bot.py
# 输入症状：我最近发烧、怕冷、一点汗都不出，浑身酸痛
```

程序会先打印召回到的片段（编号 / 病名 / 症状文本），再输出最终回答。

### 代码调用

```python
from query_bot import create_engine

engine = create_engine()
answer = engine.query("小便清长，畏寒、肢冷，可能是哪些证候")
print(answer)
```

## RAG 质量评估

```bash
python quality_eva.py
```

输出 `Recall@1/3/5/10` 与 `MRR`。评估数据在 `test_data/` 下：

- `症状文档.txt` + `病名文档.txt`：按序号对齐的评测对
- `eval_dataset.json`：含 `name / alias / query / desc` 的可溯源数据

评估数量由 `setting.py` 里的 `rag_test_len` 控制（默认 10，建议改大跑全量）。


#### 使用预先检索好的数据（可选）

也可以不重新检索，直接用预先保存好的检索结果做测试：把 `quality_eva.py` 末尾入口处的 `rag_eva_main()` 改为 `rag_eva_local_main()` 即可。

预先检索的数据在 `./retrieve_docs` 目录，包含三个文件，分别对应三种检索方式：`default`、`fusion_query`、`rerank`。三种方式的结果如下：

| 检索方式 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| default | 0.894 | 0.947 | 0.964 | 0.990 | 0.927 |
| fusion_query | 0.891 | 0.952 | 0.954 | 0.954 | 0.920 |
| rerank | **0.988** | **0.990** | **0.990** | **0.990** | **0.989** |

结论：单独使用 `rerank` 时效果最好（Recall@1 = 0.988，MRR = 0.989）；代价是 rerank 需要额外的 LLM 重排调用，检索速度和成本会更高。


## 数据说明

原始数据 `../data/demo.txt` 是《中医临床诊疗术语·证候》，每条格式为：

```
2.3.1          ← 编号
表虚证          ← 病名
因外邪袭表…所致。临床以发热…为特征的证候。   ← 病因 + 症状
```

`chroma_storage.py` 按编号切分成独立条目，以「临床」为界拆分病因与症状，**只把症状文本写入向量**（病名、别名、病因放入 metadata 且不参与向量计算）。

## 常见问题

- **检索率低**：确认已重跑 `chroma_storage.py` 重建库；确认向量里只嵌了「症状」，病名/病因没有混进去稀释信号
- **口语症状搜不准**：文档是术语（发热/恶寒/无汗），用户输入是口语（发烧/怕冷/不出汗），可在检索前加「口语 → 术语」查询改写
- **未创建集合报错**：`query_bot.py` 依赖已建好的向量库，先跑 `chroma_storage.py`
- **API Key 无效**：检查 `.env` 是否填了 `DASHSCOPE_API_KEY`，`setting.py` 里 `load_dotenv()` 会加载它

