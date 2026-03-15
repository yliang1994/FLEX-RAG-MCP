# C 阶段修改概述

## 范围

C 阶段主要完成 Ingestion Pipeline MVP，从“文档进入系统”到“生成可查询存储产物”的整条离线链路落地。目标是把加载、分块、清洗、元数据增强、图片处理、编码、索引、向量写入和脚本入口真正串起来，为 D 阶段 Retrieval MVP 提供稳定输入。

本阶段覆盖以下任务：`C1` ~ `C15`，包含核心数据契约、文件完整性检查、Loader/Chunking/Transform、Dense/Sparse 编码、存储落盘、Pipeline 编排，以及 CLI + E2E 验证。

## 阶段成果

### 1. Ingestion 核心契约与去重基座完成

- 定义了 `Document`、`Chunk`、`ChunkRecord`、`ImageRef`、`RetrievalResult` 等共享数据类型。
- 完成基于 SHA256 的文件完整性检查，可识别已成功处理的文件并跳过重复 ingestion。
- 建立了最小 `TraceContext`，为各阶段记录 stage 事件和后续 F 阶段 observability 扩展留出接口。

### 2. 文档加载、分块与 Transform 链路完成

- `PdfLoader` 已具备最小文档加载能力，支持标题提取、图片占位符解析和图片复制。
- `DocumentChunker` 已接入 `libs.splitter`，可以稳定生成 chunk id，并继承文档级 metadata。
- 完成三类 Transform：
  - `ChunkRefiner`：规则去噪 + 可选 LLM 增强 + 降级
  - `MetadataEnricher`：规则 metadata 增强 + 可选 LLM JSON 增强 + 降级
  - `ImageCaptioner`：Vision LLM caption + 非阻塞 fallback

### 3. Dense/Sparse 编码与批处理编排完成

- `DenseEncoder` 打通 `libs.embedding`，输出带 `dense_vector` 的 `ChunkRecord`。
- `SparseEncoder` 产出 BM25 友好的 `term -> normalized_tf` 稀疏表示。
- `BatchProcessor` 支持稳定分批，同时驱动 dense/sparse 编码并合并结果。

### 4. 持久化产物链路完成

- `BM25Indexer` 支持倒排索引构建、IDF 计算、JSON 持久化、查询和增量更新。
- `VectorUpserter` 支持基于来源与内容生成稳定向量 ID，并调用底层 VectorStore 做幂等 upsert。
- `ImageStorage` 支持图片文件落盘和 `image_id -> path` 的 SQLite 映射持久化。

### 5. Pipeline 与 CLI 收口完成

- `IngestionPipeline` 已串行打通：
  - `integrity -> load -> store_images -> split -> refine -> metadata -> caption -> encode -> bm25 -> vector_upsert`
- Pipeline 支持失败阶段包装、重复文件跳过和结构化 `PipelineResult` 返回。
- `scripts/ingest.py` 作为离线 CLI 入口已可运行，支持 `--path`、`--collection`、`--force`、`--config`。
- E2E 已验证首次 ingestion 产生产物，二次运行在未变更时自动跳过。

## 按任务拆分

### C1 - C3：数据契约与 Loader 基座

- `C1` 定义核心数据类型/契约
  - `ImageRef`、`Document`、`Chunk`、`ChunkRecord`、`RetrievalResult`
  - 保留兼容已有 `VectorRecord`、`QueryMatch`
- `C2` 文件完整性检查
  - `FileIntegrityChecker`
  - `SQLiteIntegrityChecker`
  - `compute_sha256` / `should_skip` / `mark_success` / `mark_failed`
- `C3` Loader 抽象基类与 PDF Loader
  - `BaseLoader.load()`
  - `PdfLoader`
  - 图片占位符解析与图片复制

### C4 - C7：Chunking 与 Transform 链路

- `C4` Splitter 集成
  - `DocumentChunker`
  - 稳定 chunk id
  - 图片 metadata 按需分发
- `C5` Transform 基类 + ChunkRefiner
  - `BaseTransform`
  - 规则去噪、可选 LLM 精炼、fallback
  - `TraceContext` 最小实现
- `C6` MetadataEnricher
  - 规则模式生成 `title/summary/tags`
  - 可选 LLM JSON 增强与降级
- `C7` ImageCaptioner
  - Vision LLM 图片 caption
  - Prompt 加载
  - 禁用/失败时非阻塞回退

### C8 - C10：编码与批处理

- `C8` DenseEncoder
  - 调用 `libs.embedding`
  - 输出 `ChunkRecord.dense_vector`
- `C9` SparseEncoder
  - 生成稀疏词权重结构
  - 空文本有明确行为
- `C10` BatchProcessor
  - 稳定分 batch
  - 合并 dense/sparse 编码结果
  - 记录批次 trace

### C11 - C13：存储落地

- `C11` BM25Indexer
  - 构建倒排索引
  - 计算 IDF
  - `index.json + docs.json` 持久化
  - 支持 load/query/upsert
- `C12` VectorUpserter
  - 生成稳定向量 ID
  - 调用 VectorStore 幂等写入
  - 保留 `original_chunk_id`
- `C13` ImageStorage
  - 保存图片文件
  - SQLite 持久化映射
  - 支持重载查回

### C14 - C15：Pipeline 与脚本入口

- `C14` Pipeline 编排
  - `IngestionPipeline`
  - 串行执行完整 ingestion 流程
  - 失败阶段包装和 skip 逻辑
- `C15` 脚本入口 ingest.py
  - CLI 参数解析
  - 调用 Pipeline
  - 输出 JSON 结果
  - E2E 验证重复运行跳过

## 主要文件

本阶段重点修改和新增的目录包括：

- [`src/core/types.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/core/types.py)
- [`src/core/settings.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/core/settings.py)
- [`src/core/trace`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/core/trace)
- [`src/libs/loader`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/libs/loader)
- [`src/ingestion/chunking`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/ingestion/chunking)
- [`src/ingestion/transform`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/ingestion/transform)
- [`src/ingestion/embedding`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/ingestion/embedding)
- [`src/ingestion/storage`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/ingestion/storage)
- [`src/ingestion/pipeline.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/src/ingestion/pipeline.py)
- [`scripts/ingest.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/scripts/ingest.py)
- [`config/prompts/chunk_refinement.txt`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/config/prompts/chunk_refinement.txt)
- [`config/prompts/image_captioning.txt`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/config/prompts/image_captioning.txt)
- [`tests/unit`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/tests/unit)
- [`tests/integration`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/tests/integration)
- [`tests/e2e/test_data_ingestion.py`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/tests/e2e/test_data_ingestion.py)

## 测试覆盖

本阶段新增或补齐的测试覆盖了：

- 核心数据类型与配置装载
- 文件完整性与状态流转
- PdfLoader 与图片占位符处理
- DocumentChunker 分块与图片 metadata 继承
- ChunkRefiner / MetadataEnricher / ImageCaptioner 的规则、LLM、fallback 路径
- DenseEncoder / SparseEncoder / BatchProcessor 的输出契约与 trace
- BM25Indexer roundtrip、查询、重建与增量更新
- VectorUpserter 幂等性
- ImageStorage 持久化与重载
- IngestionPipeline integration
- `ingest.py` CLI 的 E2E 执行与 skip 行为

阶段收口时已验证：

- `pytest -q tests/unit tests/integration/test_chroma_store_roundtrip.py tests/integration/test_chunk_refiner_llm.py tests/integration/test_ingestion_pipeline.py tests/e2e/test_data_ingestion.py`
- 结果：全部通过，含少量基于 `stub` provider 的预期跳过用例

## 提交记录

本阶段对应的主要提交如下：

1. `3b92b19` `feat(core): implement C1 shared data contracts`
2. `2563d28` `feat(loader): implement C2 file integrity checker`
3. `989cdeb` `feat(loader): implement C3 pdf loader shell`
4. `79086d2` `feat(ingestion): implement C4 document chunker`
5. `88226f2` `feat(ingestion): implement C5 chunk refiner`
6. `4155ebf` `feat(ingestion): implement C6 metadata enricher`
7. `ab43053` `feat(ingestion): implement C7 image captioner`
8. `4834055` `feat(ingestion): implement C8 dense encoder`
9. `25bd0fa` `feat(ingestion): implement C9 sparse encoder`
10. `3d69acf` `feat(ingestion): implement C10 batch processor`
11. `f8d1b41` `feat(storage): implement C11 bm25 indexer`
12. `fb323ce` `feat(storage): implement C12 vector upserter`
13. `1725faa` `feat(storage): implement C13 image storage`
14. `472427e` `feat(ingestion): implement C14 pipeline orchestration`
15. `00e2895` `feat(scripts): implement C15 ingest cli`

## 阶段结论

C 阶段完成后，项目已经具备：

- 从源文档到 chunk/metadata/vector/bm25/image 的完整离线 ingestion 链路
- 可跳过重复文件的完整性检查能力
- 可组合的 Transform/Encoder/Storage 组件
- 可验证的 BM25、向量库、图片存储持久化产物
- 可直接运行的 `ingest.py` CLI 和端到端测试

这意味着后续 D 阶段可以直接基于 C 阶段产出的向量库、BM25 索引和图片映射，进入 Query、Dense/Sparse Retrieval、Fusion 和 HybridSearch 的实现。
