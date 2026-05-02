# GuiXi (龟息) - 带宽高效的 LLM 推理框架

## 1. 项目概述

**GuiXi (龟息)** - 缓慢呼吸，高效传输。

一个用于降低 LLM 训练和推理带宽消耗的 Python 框架。通过智能压缩、语义缓存、增量同步和协议优化，在不牺牲响应质量的前提下实现 3-10 倍的带宽降低。

## 2. 功能特性

### 中文

- **Token 流压缩**：实时 LZ4/ZSTD 压缩，压缩比 3-8 倍
- **语义缓存**：基于 embedding 的相似性搜索，相似提示直接命中缓存
- **增量同步**：仅传输客户端与服务器之间的状态变更
- **二进制协议**：优化的线路格式，最小开销（12 字节头部）
- **自适应批处理**：根据网络状况动态调整批大小
- **多接口支持**：CLI、GUI (PySide6)、Web (Flask) 和 Python API
- **OpenAI 集成**：支持 LLM 智能体调用的函数工具

### English

- **Token Stream Compression**: Real-time LZ4/ZSTD compression with 3-8x ratio
- **Semantic Cache**: Embedding-based similarity search for cache hits on similar prompts
- **Delta Synchronization**: Only transmit state changes between client and server
- **Binary Protocol**: Optimized wire format with minimal overhead (12-byte header)
- **Adaptive Batching**: Dynamic batch sizing based on network conditions
- **Multi-Interface**: CLI, GUI (PySide6), Web (Flask), and Python API
- **OpenAI Integration**: Function-calling tools for LLM agent integration

## 3. 环境要求

### 中文

- **Python**: 3.9、3.10、3.11 或 3.12
- **操作系统**：Linux、macOS、Windows
- **依赖**：lz4、zstandard、numpy、websockets、flask
- **可选**：PySide6 和 pyqtgraph（用于 GUI 支持）

### English

- **Python**: 3.9, 3.10, 3.11, or 3.12
- **Operating System**: Linux, macOS, Windows
- **Dependencies**: lz4, zstandard, numpy, websockets, flask
- **Optional**: PySide6 and pyqtgraph for GUI support

## 4. 安装方法

### 从 PyPI 安装（推荐）

```bash
# 仅安装核心功能
pip install guixi

# 安装 GUI 支持
pip install guixi[gui]

# 完整开发安装
pip install guixi[all]
```

### 从源码安装

```bash
git clone https://github.com/guixi/guixi.git
cd guixi
pip install -e .
```

### 验证安装

```bash
guixi -V
python -c "import guixi; print(guixi.__version__)"
```

## 5. 快速开始

### CLI 模式

```bash
# 启动 GUI（默认）
guixi

# 从命令行运行推理
guixi infer "什么是人工智能？"

# 显示缓存统计
guixi cache --stats
```

### Python API

```python
import asyncio
from guixi import api_infer

async def main():
    result = await api_infer(prompt="什么是 AI？")
    if result.success:
        print(result.data["text"])

asyncio.run(main())
```

### Web 界面

```bash
guixi web --port 5000
# 在浏览器打开 http://localhost:5000
```

## 6. 使用说明

### CLI 命令

```bash
# 启动 GUI
guixi gui

# 启动 Web 服务器
guixi web --host 0.0.0.0 --port 8080

# 启动推理服务器
guixi server --host 0.0.0.0 --port 8080

# 运行推理
guixi infer "什么是 AI？" --max-tokens 100 --compression lz4

# 带宽基准测试
guixi bench --prompts data/prompts.txt --iterations 100

# 缓存管理
guixi cache --stats --clear
```

### CLI 标志

| 标志 | 说明 |
|------|-------------|
| `-V`, `--version` | 显示版本 |
| `-v`, `--verbose` | 详细输出 |
| `-o`, `--output` | 输出文件路径 |
| `--json` | 以 JSON 输出（使用 `dest="json_output"`） |
| `-q`, `--quiet` | 抑制非必要输出 |

## 7. Python API

### ToolResult 模式

所有 API 函数返回 `ToolResult` 数据类：

```python
from guixi import ToolResult

result = ToolResult(
    success=True,
    data={"key": "value"},
    error=None,
    metadata={"version": "0.1.0"}
)

print(result.success)    # True / False
print(result.data)       # 返回数据
print(result.error)      # 错误信息或 None
print(result.metadata)   # 元数据字典
print(result.to_dict())  # 转换为字典
```

### API 函数

```python
from guixi import api_infer, api_batch_infer, api_cache_stats

# 单次推理
result = await api_infer(
    prompt="什么是 AI？",
    max_tokens=100,
    compression="lz4",
    cache_policy="read",
)

# 批量推理
result = await api_batch_infer(
    prompts=["什么是 AI？", "什么是 ML？"],
    max_tokens=100,
    batch_size=10,
)

# 缓存统计
result = await api_cache_stats()
```

### 仅关键字参数

所有 API 函数使用仅关键字参数以提高清晰度：

```python
# 正确
result = await api_infer(prompt="test")

# 将引发 TypeError
result = await api_infer("test")
```

## 8. 智能体集成

### OpenAI 函数调用工具

GuiXi 提供 OpenAI 兼容的工具模式用于 LLM 智能体集成：

```python
from guixi import TOOLS, dispatch, list_tool_names

# 列出可用工具
print(list_tool_names())
# ['guixi_infer', 'guixi_batch_infer', 'guixi_compress', ...]

# 获取工具模式
from guixi.tools import get_tool
tool = get_tool("guixi_infer")
print(tool)

# 调度工具调用
result = dispatch("guixi_infer", {"prompt": "什么是 AI？"})
print(result)
```

### 工具模式列表

| 工具名称 | 说明 |
|-----------|-------------|
| `guixi_infer` | 运行带带宽优化的 LLM 推理 |
| `guixi_batch_infer` | 对多个提示运行批量推理 |
| `guixi_compress` | 压缩 token 序列 |
| `guixi_cache_stats` | 获取缓存统计信息 |
| `guixi_clear_cache` | 清除所有缓存条目 |
| `guixi_stream_infer` | 逐 token 流式推理 |

## 9. CLI 帮助截图

```
$ guixi --help
usage: guixi [-h] [-V] [-v] [-o OUTPUT] [--json] [-q]
            {gui,web,server,cli,bench,cache,compress,infer} ...

GuiXi (龟息) - Bandwidth-efficient LLM inference framework

options:
  -h, --help            show this help message and exit
  -V, --version         show program's version number and exit
  -v, --verbose         Verbose output
  -o OUTPUT, --output OUTPUT
                        Output file for results
  --json                 Output results as JSON
  -q, --quiet           Suppress non-essential output

Commands:
  {gui,web,server,cli,bench,cache,compress,infer}
    gui                 Launch GUI
    web                 Start web server
    server              Start inference server
    cli                 CLI mode
    bench               Benchmark bandwidth
    cache               Cache management
    compress            Compress token data
    infer               Run inference
```

## 10. 开发指南

### 项目结构

```
guixi/
├── __init__.py      # 包导出
├── __version__.py   # 版本字符串
├── __main__.py      # python -m 入口点
├── core.py          # 业务逻辑
├── cli.py           # CLI 接口
├── gui.py           # PySide6 GUI
├── app.py           # Flask Web 应用
├── api.py           # 带 ToolResult 的 Python API
├── tools.py         # OpenAI 函数调用工具
├── compress.py      # 压缩引擎
├── cache.py         # 语义缓存
├── protocol.py      # 二进制协议
└── sync.py         # 增量同步
```

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试类
pytest tests/test_unified_api.py::TestToolResult -v

# 运行覆盖率测试
pytest tests/ --cov=guixi --cov-report=term-missing
```

### 代码质量

```bash
# 格式化代码
ruff format .

# 检查 lint
ruff check .

# 类型检查
mypy guixi/
```

### 提交前检查清单

```bash
ruff format . && ruff check . && mypy . && pytest
```

## 11. 许可证

GuiXi 根据 **GNU General Public License v3.0 (GPLv3)** 发布。

这意味着您可以：
- 以任何目的使用此软件
- 修改和分发源代码
- 在动态链接的应用程序中使用该库

但须遵守以下条件：
- 修改必须以 GPLv3 发布
- 必须提供修改的完整源代码
- 必须保留归属

如需商业许可咨询，请联系 team@guixi.dev。

### 第三方许可证

- LZ4 - BSD 许可证
- Zstandard - BSD 许可证
- NumPy - BSD 许可证
- WebSockets - BSD 许可证
- Flask - BSD 许可证
- PySide6 - LGPL 许可证

---

**GuiXi (龟息)** - 缓慢呼吸，高效传输。
