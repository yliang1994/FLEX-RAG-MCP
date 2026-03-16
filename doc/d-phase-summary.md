# D 阶段修改概述

## 范围

D 阶段主要完成 Retrieval MVP，从“用户输入 query”到“返回可排序的 Top-K chunk 候选”的在线查询链路落地。目标是把 Query 预处理、Dense / Sparse 检索、RRF 融合、Hybrid 编排、Rerank 回退机制和查询 CLI 串起来，为后续 E 阶段 MCP Tool 直接复用。

本阶段覆盖以下任务：`D1` ~ `D7`，包含查询契约、QueryProcessor、DenseRetriever、SparseRetriever、Fusion、HybridSearch、Core Reranker 和 `query.py` 命令行入口。

## 阶段成果

### 1. Query 预处理与检索契约完成

- 在 [`src/core/types.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/core/types.py) 中补充了 `ProcessedQuery`，用于承载原始 query、归一化 query、关键词列表和 metadata filters。
- 在 [`src/core/query_engine/query_processor.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/core/query_engine/query_processor.py) 中实现了规则版 `QueryProcessor`：
  - 支持关键词提取、去重和小写归一化；
  - 支持从 query 中解析 `collection/doc_type/language/access_level` 四类内联 filter；
  - 在关键词提取为空时回退到归一化 query。

### 2. Dense / Sparse 检索链路落地

- 在 [`src/core/query_engine/dense_retriever.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/core/query_engine/dense_retriever.py) 中实现了 `DenseRetriever`：
  - 调用 `EmbeddingFactory` 生成 query embedding；
  - 调用 `VectorStore.query()` 执行向量检索；
  - 将结果规范化为 `RetrievalResult`；
  - 支持依赖注入和 trace 记录。
- 在 [`src/core/query_engine/sparse_retriever.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/core/query_engine/sparse_retriever.py) 中实现了 `SparseRetriever`：
  - 使用 `BM25Indexer.search()` 对关键词进行稀疏召回；
  - 通过 `VectorStore.get_by_ids()` 回查 chunk 的 `text` 和 `metadata`；
  - 把 BM25 分数与 payload 合并为 `RetrievalResult`。
- 为支撑 Sparse 链路，在 [`src/libs/vector_store/base_vector_store.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/libs/vector_store/base_vector_store.py) 和 [`src/libs/vector_store/chroma_store.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/libs/vector_store/chroma_store.py) 中新增并实现了 `get_by_ids()`。

### 3. Fusion / Hybrid / Rerank 编排完成

- 在 [`src/core/query_engine/fusion.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/core/query_engine/fusion.py) 中实现了 `RRFusion`：
  - 支持 configurable `k`；
  - 支持 dense / sparse 两路排序融合；
  - 对重复 `chunk_id` 做去重，并保持 deterministic 排序。
- 在 [`src/core/query_engine/hybrid_search.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/core/query_engine/hybrid_search.py) 中实现了 `HybridSearch`：
  - 调用 `QueryProcessor.process()` 生成 `ProcessedQuery`；
  - 编排 DenseRetriever、SparseRetriever 与 RRFusion；
  - 支持 metadata 后置过滤；
  - Dense / Sparse 任一路失败时自动降级到单路结果；
  - 记录各阶段的 trace 摘要。
- 在 [`src/core/query_engine/reranker.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/core/query_engine/reranker.py) 中实现了 Core 层 `Reranker` 适配器：
  - 把 `RetrievalResult` 转换成 libs 层 `QueryMatch`；
  - 调用 `RerankerFactory` 解析的后端；
  - 后端异常或返回非法 candidate ids 时，稳定回退到原始 fusion 排序；
  - 通过 `last_fallback` / `last_fallback_reason` 和 trace 标记回退状态。

### 4. 查询 CLI 收口完成

- 新增 [`scripts/query.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/scripts/query.py)，作为在线查询的 CLI 入口：
  - 支持 `--query`、`--top-k`、`--collection`、`--verbose`、`--no-rerank`、`--config`；
  - 复用 `HybridSearch + Reranker` 主链路；
  - 默认输出 Top-K 结果（score、source、page、chunk_id、文本摘要）；
  - `--verbose` 模式输出 trace 阶段明细；
  - 当本地没有已摄取数据时，返回友好提示而不是堆栈异常。

## 按任务拆分

### D1：QueryProcessor

- 新增 `ProcessedQuery` 契约；
- 实现规则版 query 归一化、关键词提取和 filter 解析；
- 新增：
  - [`tests/unit/test_query_processor.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/tests/unit/test_query_processor.py)
  - [`tests/unit/test_core_types.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/tests/unit/test_core_types.py) 中的 `ProcessedQuery` 序列化覆盖

### D2：DenseRetriever

- 实现 `query -> embed -> vector_store.query -> RetrievalResult` 编排；
- 支持依赖注入、filter 透传和 trace；
- 新增 [`tests/unit/test_dense_retriever.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/tests/unit/test_dense_retriever.py)

### D3：SparseRetriever

- 实现 `keywords -> BM25 -> vector_store.get_by_ids -> RetrievalResult` 编排；
- 扩展 VectorStore 契约支持 `get_by_ids()`；
- 新增 [`tests/unit/test_sparse_retriever.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/tests/unit/test_sparse_retriever.py)
- 更新：
  - [`tests/unit/test_vector_store_contract.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/tests/unit/test_vector_store_contract.py)
  - [`tests/unit/test_dense_retriever.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/tests/unit/test_dense_retriever.py)
  - [`tests/unit/test_vector_upserter_idempotency.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/tests/unit/test_vector_upserter_idempotency.py)

### D4：RRF Fusion

- 实现 `RRFusion.fuse()`；
- 支持重复 chunk 去重、可配置 k 和 top_k 裁剪；
- 新增 [`tests/unit/test_fusion_rrf.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/tests/unit/test_fusion_rrf.py)

### D5：HybridSearch

- 实现 QueryProcessor、DenseRetriever、SparseRetriever、RRFusion 的主链路编排；
- 支持 metadata filter 和单路降级；
- 新增 [`tests/integration/test_hybrid_search.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/tests/integration/test_hybrid_search.py)

### D6：Core Reranker

- 实现 libs 层 reranker 到 core 层结果契约的适配；
- 支持 backend exception / invalid ids 双路径 fallback；
- 新增 [`tests/unit/test_reranker_fallback.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/tests/unit/test_reranker_fallback.py)

### D7：query.py CLI

- 新增在线查询命令行入口 [`scripts/query.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/scripts/query.py)；
- 打通 `HybridSearch + Reranker` 查询流程；
- 支持 verbose trace 输出和无数据友好提示。

## 主要文件

本阶段重点修改和新增的目录包括：

- [`src/core/types.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/core/types.py)
- [`src/core/query_engine`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/core/query_engine)
- [`src/libs/vector_store/base_vector_store.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/libs/vector_store/base_vector_store.py)
- [`src/libs/vector_store/chroma_store.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/libs/vector_store/chroma_store.py)
- [`scripts/query.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/scripts/query.py)
- [`tests/unit`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/tests/unit)
- [`tests/integration/test_hybrid_search.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/tests/integration/test_hybrid_search.py)

## 测试与验证

本阶段按任务逐步补齐并执行了以下验证：

- `pytest -q tests/unit/test_query_processor.py tests/unit/test_core_types.py`
- `pytest -q tests/unit/test_dense_retriever.py tests/unit/test_query_processor.py tests/unit/test_core_types.py`
- `pytest -q tests/unit/test_sparse_retriever.py tests/unit/test_vector_store_contract.py tests/unit/test_dense_retriever.py tests/unit/test_vector_upserter_idempotency.py tests/integration/test_chroma_store_roundtrip.py`
- `pytest -q tests/unit/test_fusion_rrf.py tests/unit/test_dense_retriever.py tests/unit/test_sparse_retriever.py`
- `pytest -q tests/integration/test_hybrid_search.py tests/unit/test_query_processor.py tests/unit/test_fusion_rrf.py`
- `pytest -q tests/unit/test_reranker_fallback.py tests/unit/test_llm_reranker.py tests/unit/test_cross_encoder_reranker.py`
- `python -m compileall scripts/query.py src/core/query_engine`
- `python scripts/query.py --query "测试查询" --verbose`

阶段收口时，D 阶段相关测试与 CLI 验证均通过。

## 提交记录

本阶段对应的主要提交如下：

1. `14236ed` `feat(retrieval): implement query processor`
2. `6651198` `feat(retrieval): add dense retriever`
3. `783577b` `feat(retrieval): add sparse retriever`
4. `cfffc98` `feat(retrieval): add rrf fusion`
5. `62781b7` `feat(retrieval): add hybrid search orchestration`
6. `307db93` `feat(retrieval): add reranker fallback orchestration`
7. 当前提交收口 D7 CLI 与阶段文档

## 阶段结论

D 阶段完成后，项目已经具备：

- 从自然语言 query 到可排序 Top-K 候选的完整 Retrieval 链路；
- 规则版 QueryProcessor、Dense / Sparse 双路召回与 RRF 融合能力；
- 可降级的 HybridSearch 和 Core Reranker；
- 可直接运行的 `query.py` CLI，用于本地查询调试和后续 MCP Tool 复用；
- 为 E 阶段 MCP Server Tools 提供稳定的核心查询入口。

这意味着后续 E 阶段可以直接围绕现有 `HybridSearch + Reranker + query.py` 逻辑封装 MCP Tools，而不必再重复实现检索主链路。
