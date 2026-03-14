# B 阶段修改概述

## 范围

B 阶段主要完成 `clean-start` 分支中 Libs 可插拔层与默认实现的第一轮落地，目标是把“抽象接口 + 工厂路由 + 默认后端 + 测试基座”补齐，为后续 C 阶段的 Ingestion Pipeline 和 D/E/F 阶段的 Retrieval、MCP、Tracing 提供稳定底座。

本阶段覆盖以下任务：`B1` ~ `B9`，其中包含抽象接口、默认 provider/stub、Vision 扩展、持久化 roundtrip，以及多组 smoke/unit/integration 测试。

## 阶段成果

### 1. 抽象层与工厂模式落地

- 完成了 LLM、Embedding、Splitter、VectorStore、Reranker、Evaluator 六类基础抽象层。
- 每一层都实现了统一的 `Factory` 路由机制，支持通过配置切换 provider/backend。
- 为测试和后续扩展保留了 `register()` 式动态注册入口，支持注入 Fake/Mock 实现。

### 2. 默认实现补齐

- **LLM**
  - OpenAI / Azure / DeepSeek provider 补齐了输入校验与可读错误。
  - Ollama provider 支持 `base_url`、`timeout`、可替换 `transport`，可在无真实网络环境下测试。
  - Vision 能力完成抽象接口、工厂集成和 Azure Vision mock 实现。
- **Embedding**
  - OpenAI / Azure / Ollama embedding provider 均支持明确的输入校验与 deterministic/mock 运行路径。
  - Azure embedding 复用了 OpenAI embedding 的核心校验逻辑。
- **Splitter**
  - Splitter 配置接入 `Settings`，新增 `splitter.method/chunk_size/chunk_overlap`。
  - 默认 `RecursiveSplitter` 已升级为 Markdown block-aware 实现，可保留标题和 fenced code block。
- **Vector Store**
  - `ChromaStore` 从内存 stub 升级为最小 JSON 持久化实现，支持 `upsert -> query` roundtrip。
- **Reranker**
  - `NoneReranker`、`LLMReranker`、`CrossEncoderReranker` 全部可跑通。
  - LLM Reranker 支持读取 `config/prompts/rerank.txt`、结构化 `ranked_ids` 校验和 fallback signal。
  - Cross-Encoder Reranker 支持 mock scorer、backend alias 和失败回退信号。
- **Evaluator**
  - `CustomEvaluator` 实现了 `hit_rate`、`mrr`、`precision_at_k`、`recall`。
  - `RagasEvaluator` 补齐了最小占位实现，保证接口稳定。

### 3. 核心契约补齐

- 在 [`src/core/types.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/core/types.py) 中补充了 `VectorRecord`、`QueryMatch` 等共享契约，支撑 VectorStore 和 Reranker 等链路复用。
- 在 [`src/core/settings.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/core/settings.py) 和 [`config/settings.yaml`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/config/settings.yaml) 中扩展了 `splitter` 配置，保证工厂与配置对齐。

### 4. 测试覆盖扩张

- 新增或完善的测试覆盖了：
  - LLM / Vision LLM factory 与 provider smoke
  - Embedding factory 与 provider smoke
  - Splitter factory 与 recursive splitter 行为
  - VectorStore contract 与 Chroma roundtrip integration
  - Reranker factory / LLM reranker / Cross-Encoder reranker
  - Evaluator factory 与 CustomEvaluator 指标计算
- 当前阶段结束时，已验证：
  - `pytest -q tests/unit tests/integration/test_chroma_store_roundtrip.py`
  - 结果：`72/72` 通过

## 按任务拆分

### B1 - B6：工厂与抽象接口基座

- `B1` LLM 抽象接口与工厂
  - `BaseLLM`、`ChatMessage`、`LLMResponse`
  - `LLMFactory.create()` 与 provider 注册机制
- `B2` Embedding 抽象接口与工厂
  - `BaseEmbedding`
  - `EmbeddingFactory.create()` 与 provider 注册机制
- `B3` Splitter 抽象接口与工厂
  - `BaseSplitter`
  - `SplitterFactory.create()`
  - `recursive` / `semantic` / `fixed` 三种默认 splitter
- `B4` VectorStore 抽象接口与工厂
  - `BaseVectorStore.upsert/query`
  - `VectorStoreFactory.create()`
  - `VectorRecord` / `QueryMatch` 契约
- `B5` Reranker 抽象接口与工厂
  - `BaseReranker`
  - `NoneReranker`
  - `RerankerFactory.create()`
- `B6` Evaluator 抽象接口与工厂
  - `BaseEvaluator`
  - `EvaluatorFactory.create()`
  - `CustomEvaluator` / `RagasEvaluator`

### B7：默认实现补齐

- `B7.1` OpenAI-compatible LLM
  - OpenAI / Azure / DeepSeek 的消息校验、清晰报错与 smoke test
- `B7.2` Ollama LLM
  - 本地 endpoint、timeout、transport 注入、错误分类
- `B7.3` OpenAI & Azure Embedding
  - 输入校验、Azure 特有配置、行为一致性
- `B7.4` Ollama Embedding
  - 批量 embed、transport 注入、错误分类
- `B7.5` Recursive Splitter
  - Markdown block-aware 切分，保留标题与代码块
- `B7.6` ChromaStore
  - JSON 持久化与 roundtrip integration
- `B7.7` LLM Reranker
  - prompt 读取、结构化 `ranked_ids`、fallback signal
- `B7.8` Cross-Encoder Reranker
  - mock scorer、alias 路由、fallback signal

### B8 - B9：Vision LLM 能力接入

- `B8` Vision LLM 抽象接口与工厂集成
  - `BaseVisionLLM`
  - `VisionInput` / `VisionResponse`
  - `LLMFactory.create_vision_llm()` / `register_vision()`
- `B9` Azure Vision LLM
  - 支持路径 / base64 / bytes 输入
  - 支持 `azure_endpoint` / `api_version` / `deployment_name`
  - 支持最小图片预处理和清晰错误分类

## 主要文件

本阶段重点修改和新增的目录包括：

- [`src/libs/llm`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/libs/llm)
- [`src/libs/embedding`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/libs/embedding)
- [`src/libs/splitter`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/libs/splitter)
- [`src/libs/vector_store`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/libs/vector_store)
- [`src/libs/reranker`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/libs/reranker)
- [`src/libs/evaluator`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/libs/evaluator)
- [`src/core/settings.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/core/settings.py)
- [`src/core/types.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/core/types.py)
- [`config/settings.yaml`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/config/settings.yaml)
- [`config/prompts/rerank.txt`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/config/prompts/rerank.txt)
- [`tests/unit`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/tests/unit)
- [`tests/integration/test_chroma_store_roundtrip.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/tests/integration/test_chroma_store_roundtrip.py)

## 提交记录

本阶段对应的主要提交如下：

1. `c83dfac` `feat(llm): implement B1 llm factory`
2. `dd2f678` `feat(embedding): implement B2 embedding factory`
3. `a207a4f` `feat(splitter): implement B3 splitter factory`
4. `46af3ed` `feat(vector-store): implement B4 vector store contract`
5. `c1f8a85` `feat(reranker): implement B5 reranker factory`
6. `e46011b` `feat(evaluator): implement B6 evaluator factory`
7. `7fede01` `feat(llm): implement B7.1 openai compatible providers`
8. `f3c82bb` `feat(llm): implement B7.2 ollama llm`
9. `0ab4d89` `feat(embedding): implement B7.3 openai azure embeddings`
10. `985506c` `feat(embedding): implement B7.4 ollama embedding`
11. `60b14c9` `feat(splitter): implement B7.5 recursive splitter`
12. `b91da36` `feat(vector-store): implement B7.6 chroma roundtrip`
13. `aa40564` `feat(reranker): implement B7.7 llm reranker`
14. `7dbfa6f` `feat(reranker): implement B7.8 cross encoder reranker`
15. `e069e82` `feat(llm): implement B8 vision llm factory`
16. `fc6ff94` `feat(llm): implement B9 azure vision llm`

## 阶段结论

B 阶段完成后，项目已经具备：

- 可配置的 Libs 插件层
- 可测试的默认 provider / backend / stub
- Vision LLM 抽象和 Azure Vision mock 能力
- 持久化的 Chroma roundtrip 验证路径
- 稳定的测试基座，可直接支撑阶段 C 的 Ingestion 与后续 Retrieval/MCP 开发

这意味着后续阶段可以直接基于这些抽象与默认实现，开始把文档摄取、检索、MCP Tool、Trace 与 Dashboard 串成完整链路。
