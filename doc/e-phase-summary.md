# E 阶段修改概述

## 范围

E 阶段主要完成 MCP Server 层与对外 Tools 落地，把 D 阶段已经具备的 Retrieval 能力封装成标准 MCP 能调用的接口。目标是打通 `stdio server -> JSON-RPC 协议处理 -> tools/list -> tools/call -> 文本/图像响应` 这条完整对外链路。

本阶段覆盖以下任务：`E1` ~ `E6`，包含 MCP Server 入口、协议处理层、`query_knowledge_hub`、`list_collections`、`get_document_summary` 以及多模态返回组装。

## 阶段成果

### 1. MCP Server 入口与协议主循环完成

- 在 [`src/mcp_server/server.py`](/home/yliang/cv_project/FLEX-RAG-MCP/src/mcp_server/server.py) 中实现了最小可运行的 stdio MCP server：
  - 使用 `Content-Length` framing 收发 JSON-RPC 消息；
  - 保证 `stdout` 只输出协议消息；
  - 日志输出走 `stderr`，满足 MCP transport 约束；
  - 支持 `initialize`、`notifications/initialized`、`exit` 的主循环行为。
- 新增 [`tests/integration/test_mcp_server.py`](/home/yliang/cv_project/FLEX-RAG-MCP/tests/integration/test_mcp_server.py) 子进程集成测试，验证 initialize 握手与 stdout/stderr 隔离。

### 2. JSON-RPC 协议处理层完成

- 在 [`src/mcp_server/protocol_handler.py`](/home/yliang/cv_project/FLEX-RAG-MCP/src/mcp_server/protocol_handler.py) 中实现了 `ProtocolHandler`：
  - 统一处理 `initialize`、`tools/list`、`tools/call`；
  - 定义 `ToolDefinition` 作为 tool schema 与执行函数的注册单元；
  - 实现 JSON-RPC 错误码映射：
    - `-32600` Invalid Request
    - `-32601` Method not found
    - `-32602` Invalid params
    - `-32603` Internal error
  - 隐藏内部异常细节，避免堆栈泄露给 MCP Client。
- 新增 [`tests/unit/test_protocol_handler.py`](/home/yliang/cv_project/FLEX-RAG-MCP/tests/unit/test_protocol_handler.py)：
  - 覆盖 initialize/serverInfo/capabilities；
  - 覆盖 tools/list schema 返回；
  - 覆盖 tools/call 路由；
  - 覆盖无效方法、参数错误、内部异常隐藏。

### 3. `query_knowledge_hub` 工具完成

- 在 [`src/mcp_server/tools/query_knowledge_hub.py`](/home/yliang/cv_project/FLEX-RAG-MCP/src/mcp_server/tools/query_knowledge_hub.py) 中实现了主检索 tool：
  - 调用 `HybridSearch` 执行召回；
  - 调用 `Reranker` 执行重排；
  - 调用 `ResponseBuilder` 生成 MCP 响应。
- 在 [`src/core/response/citation_generator.py`](/home/yliang/cv_project/FLEX-RAG-MCP/src/core/response/citation_generator.py) 中实现结构化 citations 生成：
  - 输出 `source` / `page` / `chunk_id` / `score` / `index`。
- 在 [`src/core/response/response_builder.py`](/home/yliang/cv_project/FLEX-RAG-MCP/src/core/response/response_builder.py) 中实现 MCP 文本响应构建：
  - `content[0]` 输出带 `[1]`、`[2]` 标号的 Markdown；
  - `structuredContent` 输出 query、citations 与 result_count；
  - 无结果时返回友好提示，而不是空数组或异常。
- 新增 [`tests/unit/test_response_builder.py`](/home/yliang/cv_project/FLEX-RAG-MCP/tests/unit/test_response_builder.py)；
- 扩展 [`tests/integration/test_mcp_server.py`](/home/yliang/cv_project/FLEX-RAG-MCP/tests/integration/test_mcp_server.py)，验证 `tools/call query_knowledge_hub` 的空结果响应路径。

### 4. `list_collections` 工具完成

- 在 [`src/mcp_server/tools/list_collections.py`](/home/yliang/cv_project/FLEX-RAG-MCP/src/mcp_server/tools/list_collections.py) 中实现本地集合扫描：
  - 默认扫描 `data/documents/`；
  - 支持注入根目录，便于测试；
  - 返回集合名、目录路径、文档数统计；
  - 无集合时返回友好提示。
- 在 [`src/mcp_server/tools/__init__.py`](/home/yliang/cv_project/FLEX-RAG-MCP/src/mcp_server/tools/__init__.py) 中完成 tool 注册，`tools/list` 可以暴露 `list_collections` schema。
- 新增 [`tests/unit/test_list_collections.py`](/home/yliang/cv_project/FLEX-RAG-MCP/tests/unit/test_list_collections.py)：
  - 覆盖排序后的集合列表；
  - 覆盖空目录友好返回。

### 5. `get_document_summary` 工具完成

- 在 [`src/mcp_server/tools/get_document_summary.py`](/home/yliang/cv_project/FLEX-RAG-MCP/src/mcp_server/tools/get_document_summary.py) 中实现按 `doc_id` 读取本地摘要：
  - 默认从 `data/db/chroma/records.json` 查找；
  - 优先读取 metadata 中的 `title/summary/tags`；
  - metadata 缺失时从文本内容回退生成摘要；
  - 不存在的 `doc_id` 返回规范参数错误。
- 在 [`src/mcp_server/tools/__init__.py`](/home/yliang/cv_project/FLEX-RAG-MCP/src/mcp_server/tools/__init__.py) 中为该工具增加参数包装，把 `ValueError` 转成 `InvalidParamsError`。
- 新增 [`tests/unit/test_get_document_summary.py`](/home/yliang/cv_project/FLEX-RAG-MCP/tests/unit/test_get_document_summary.py)：
  - 覆盖正常返回；
  - 覆盖 metadata 缺失时的摘要回退；
  - 覆盖缺失 doc_id 的错误分支。

### 6. 多模态返回组装完成

- 在 [`src/core/response/multimodal_assembler.py`](/home/yliang/cv_project/FLEX-RAG-MCP/src/core/response/multimodal_assembler.py) 中实现 `MultimodalAssembler`：
  - 从 `RetrievalResult.metadata.images` 提取图片路径；
  - 读取图片文件并转为 base64；
  - 自动推断 `mimeType`；
  - 去重同一路径图片，避免重复返回。
- 更新 [`src/core/response/response_builder.py`](/home/yliang/cv_project/FLEX-RAG-MCP/src/core/response/response_builder.py)：
  - 在文本 content 后追加 image content；
  - 保持 `structuredContent.citations` 结构不变。
- 更新 [`tests/unit/test_response_builder.py`](/home/yliang/cv_project/FLEX-RAG-MCP/tests/unit/test_response_builder.py)，覆盖 image content 追加；
- 更新 [`tests/integration/test_mcp_server.py`](/home/yliang/cv_project/FLEX-RAG-MCP/tests/integration/test_mcp_server.py)，通过临时工作目录和伪造 `records.json` 验证真实 `tools/call` 路径返回图片。

## 按任务拆分

### E1：MCP Server 入口与 Stdio 约束

- 实现 `MCPServer` 主循环、消息 framing、stderr 日志隔离；
- 新增子进程集成测试验证 initialize 与 stdout clean。

### E2：Protocol Handler 协议解析与能力协商

- 实现 `ProtocolHandler` 与 `ToolDefinition`；
- 完成 `initialize` / `tools/list` / `tools/call`；
- 完成 JSON-RPC 错误处理与测试。

### E3：`query_knowledge_hub`

- 实现检索工具主链路；
- 新增 citations 与 MCP 文本响应构建；
- 打通 `tools/call` 到真实检索逻辑。

### E4：`list_collections`

- 实现集合目录扫描与文档计数；
- 注册到 MCP tool 列表；
- 补齐单测与 tool schema 覆盖。

### E5：`get_document_summary`

- 实现按 `doc_id` 的本地摘要查询；
- 支持 metadata 回退；
- 将不存在文档映射为规范参数错误。

### E6：多模态返回组装

- 实现图片 base64 编码返回；
- 在 `query_knowledge_hub` 响应中追加 image content；
- 新增图像集成测试。

## 主要文件

本阶段重点修改和新增的文件包括：

- [`src/mcp_server/server.py`](/home/yliang/cv_project/FLEX-RAG-MCP/src/mcp_server/server.py)
- [`src/mcp_server/protocol_handler.py`](/home/yliang/cv_project/FLEX-RAG-MCP/src/mcp_server/protocol_handler.py)
- [`src/mcp_server/tools/__init__.py`](/home/yliang/cv_project/FLEX-RAG-MCP/src/mcp_server/tools/__init__.py)
- [`src/mcp_server/tools/query_knowledge_hub.py`](/home/yliang/cv_project/FLEX-RAG-MCP/src/mcp_server/tools/query_knowledge_hub.py)
- [`src/mcp_server/tools/list_collections.py`](/home/yliang/cv_project/FLEX-RAG-MCP/src/mcp_server/tools/list_collections.py)
- [`src/mcp_server/tools/get_document_summary.py`](/home/yliang/cv_project/FLEX-RAG-MCP/src/mcp_server/tools/get_document_summary.py)
- [`src/core/response/citation_generator.py`](/home/yliang/cv_project/FLEX-RAG-MCP/src/core/response/citation_generator.py)
- [`src/core/response/response_builder.py`](/home/yliang/cv_project/FLEX-RAG-MCP/src/core/response/response_builder.py)
- [`src/core/response/multimodal_assembler.py`](/home/yliang/cv_project/FLEX-RAG-MCP/src/core/response/multimodal_assembler.py)
- [`tests/unit/test_protocol_handler.py`](/home/yliang/cv_project/FLEX-RAG-MCP/tests/unit/test_protocol_handler.py)
- [`tests/unit/test_response_builder.py`](/home/yliang/cv_project/FLEX-RAG-MCP/tests/unit/test_response_builder.py)
- [`tests/unit/test_list_collections.py`](/home/yliang/cv_project/FLEX-RAG-MCP/tests/unit/test_list_collections.py)
- [`tests/unit/test_get_document_summary.py`](/home/yliang/cv_project/FLEX-RAG-MCP/tests/unit/test_get_document_summary.py)
- [`tests/integration/test_mcp_server.py`](/home/yliang/cv_project/FLEX-RAG-MCP/tests/integration/test_mcp_server.py)

## 测试与验证

E 阶段逐步补齐并执行了以下验证：

- `pytest -q tests/integration/test_mcp_server.py`
- `pytest -q tests/unit/test_protocol_handler.py`
- `pytest -q tests/unit/test_response_builder.py`
- `pytest -q tests/unit/test_list_collections.py`
- `pytest -q tests/unit/test_get_document_summary.py`
- `pytest -q tests/integration/test_mcp_server.py -k query_knowledge_hub`
- `pytest -q tests/integration/test_mcp_server.py -k image`

阶段收口时，`tests/integration/test_mcp_server.py` 全量通过，协议层与响应层单测均通过。

## 提交记录

本阶段对应的主要提交如下：

1. `e0809f2` `feat(mcp): add protocol handler and stdio bootstrap`
2. `ff0f706` `feat(mcp): implement query knowledge hub tool`
3. `07ee787` `feat(mcp): implement list collections tool`
4. `6658b4f` `feat(mcp): implement document summary tool`
5. `f3d1114` `feat(mcp): implement multimodal response assembly`

## 阶段结论

E 阶段完成后，项目已经具备：

- 可运行的本地 MCP stdio server；
- 完整的 JSON-RPC 2.0 协议处理与 tool 注册能力；
- `query_knowledge_hub`、`list_collections`、`get_document_summary` 三个可调用 MCP tools；
- 文本 citations 与图片 base64 的多模态 MCP 响应；
- 可由 Copilot / Claude 一类 MCP Client 直接消费的基础服务层。

这意味着后续 F 阶段可以直接在现有 MCP 主链路上补 Trace 基础设施与打点，而不需要再回头重构协议层或响应层。
