# GuiXi (龟息) - Bandwidth-Efficient LLM Inference Framework

## 1. Project Overview

**GuiXi (龟息)** - Breathe slowly, transmit efficiently.

A Python framework for reducing bandwidth in LLM training and inference through intelligent compression, semantic caching, delta synchronization, and protocol optimization. Achieve 3-10x bandwidth reduction without sacrificing response quality.

## 2. Features

### English

- **Token Stream Compression**: Real-time LZ4/ZSTD compression with 3-8x ratio
- **Semantic Cache**: Embedding-based similarity search for cache hits on similar prompts
- **Delta Synchronization**: Only transmit state changes between client and server
- **Binary Protocol**: Optimized wire format with minimal overhead (12-byte header)
- **Adaptive Batching**: Dynamic batch sizing based on network conditions
- **Multi-Interface**: CLI, GUI (PySide6), Web (Flask), and Python API
- **OpenAI Integration**: Function-calling tools for LLM agent integration

### 中文

- **Token 流压缩**：实时 LZ4/ZSTD 压缩，压缩比 3-8 倍
- **语义缓存**：基于 embedding 的相似性搜索，相似提示直接命中缓存
- **增量同步**：仅传输客户端与服务器之间的状态变更
- **二进制协议**：优化的线路格式，最小开销（12 字节头部）
- **自适应批处理**：根据网络状况动态调整批大小
- **多接口支持**：CLI、GUI (PySide6)、Web (Flask) 和 Python API
- **OpenAI 集成**：支持 LLM 智能体调用的函数工具

## 3. Requirements

### English

- **Python**: 3.9, 3.10, 3.11, or 3.12
- **Operating System**: Linux, macOS, Windows
- **Dependencies**: lz4, zstandard, numpy, websockets, flask
- **Optional**: PySide6 and pyqtgraph for GUI support

### 中文

- **Python**: 3.9、3.10、3.11 或 3.12
- **操作系统**：Linux、macOS、Windows
- **依赖**：lz4、zstandard、numpy、websockets、flask
- **可选**：PySide6 和 pyqtgraph（用于 GUI 支持）

## 4. Installation

### From PyPI (Recommended)

```bash
# Core only
pip install guixi

# With GUI support
pip install guixi[gui]

# Full development installation
pip install guixi[all]
```

### From Source

```bash
git clone https://github.com/guixi/guixi.git
cd guixi
pip install -e .
```

### Verify Installation

```bash
guixi -V
python -c "import guixi; print(guixi.__version__)"
```

## 5. Quick Start

### CLI Mode

```bash
# Launch GUI (default)
guixi

# Run inference from command line
guixi infer "What is artificial intelligence?"

# Show cache statistics
guixi cache --stats
```

### Python API

```python
import asyncio
from guixi import api_infer

async def main():
    result = await api_infer(prompt="What is AI?")
    if result.success:
        print(result.data["text"])

asyncio.run(main())
```

### Web Interface

```bash
guixi web --port 5000
# Open http://localhost:5000 in browser
```

## 6. Usage

### CLI Commands

```bash
# Launch GUI
guixi gui

# Start web server
guixi web --host 0.0.0.0 --port 8080

# Start inference server
guixi server --host 0.0.0.0 --port 8080

# Run inference
guixi infer "What is AI?" --max-tokens 100 --compression lz4

# Benchmark bandwidth
guixi bench --prompts data/prompts.txt --iterations 100

# Cache management
guixi cache --stats --clear
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `-V`, `--version` | Show version |
| `-v`, `--verbose` | Verbose output |
| `-o`, `--output` | Output file path |
| `--json` | Output as JSON (use `dest="json_output"`) |
| `-q`, `--quiet` | Suppress non-essential output |

## 7. Python API

### ToolResult Pattern

All API functions return a `ToolResult` dataclass:

```python
from guixi import ToolResult

result = ToolResult(
    success=True,
    data={"key": "value"},
    error=None,
    metadata={"version": "0.1.0"}
)

print(result.success)    # True / False
print(result.data)       # Return data
print(result.error)      # Error message or None
print(result.metadata)   # Metadata dict
print(result.to_dict())  # Convert to dict
```

### API Functions

```python
from guixi import api_infer, api_batch_infer, api_cache_stats

# Single inference
result = await api_infer(
    prompt="What is AI?",
    max_tokens=100,
    compression="lz4",
    cache_policy="read",
)

# Batched inference
result = await api_batch_infer(
    prompts=["What is AI?", "What is ML?"],
    max_tokens=100,
    batch_size=10,
)

# Cache statistics
result = await api_cache_stats()
```

### Keyword-Only Arguments

All API functions use keyword-only arguments for clarity:

```python
# Correct
result = await api_infer(prompt="test")

# Will raise TypeError
result = await api_infer("test")
```

## 8. Agent Integration

### OpenAI Function-Calling Tools

GuiXi provides OpenAI-compatible tool schemas for LLM agent integration:

```python
from guixi import TOOLS, dispatch, list_tool_names

# List available tools
print(list_tool_names())
# ['guixi_infer', 'guixi_batch_infer', 'guixi_compress', ...]

# Get tool schema
from guixi.tools import get_tool
tool = get_tool("guixi_infer")
print(tool)

# Dispatch tool call
result = dispatch("guixi_infer", {"prompt": "What is AI?"})
print(result)
```

### Tool Schemas

| Tool Name | Description |
|-----------|-------------|
| `guixi_infer` | Run LLM inference with bandwidth optimization |
| `guixi_batch_infer` | Run batched inference for multiple prompts |
| `guixi_compress` | Compress a token sequence |
| `guixi_cache_stats` | Get cache statistics |
| `guixi_clear_cache` | Clear all cached entries |
| `guixi_stream_infer` | Stream inference results token by token |

## 9. CLI Help Screenshot

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

## 10. Development

### Project Structure

```
guixi/
├── __init__.py      # Package exports
├── __version__.py   # Version string
├── __main__.py      # python -m entry point
├── core.py          # Business logic
├── cli.py           # CLI interface
├── gui.py           # PySide6 GUI
├── app.py           # Flask web app
├── api.py           # Python API with ToolResult
├── tools.py         # OpenAI function-calling tools
├── compress.py      # Compression engine
├── cache.py         # Semantic cache
├── protocol.py      # Binary protocol
└── sync.py         # Delta synchronization
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_unified_api.py::TestToolResult -v

# Run with coverage
pytest tests/ --cov=guixi --cov-report=term-missing
```

### Code Quality

```bash
# Format code
ruff format .

# Check linting
ruff check .

# Type checking
mypy guixi/
```

### Pre-Commit Checklist

```bash
ruff format . && ruff check . && mypy . && pytest
```

## 11. License

GuiXi is released under the **GNU General Public License v3.0 (GPLv3)**.

This means you are free to:
- Use this software for any purpose
- Modify and distribute the source code
- Use the library in proprietary applications (linked dynamically)

Under the following conditions:
- Modifications must be released under GPLv3
- Full source code of modifications must be available
- Attribution must be maintained

For commercial licensing inquiries, contact team@guixi.dev.

### Third-Party Licenses

- LZ4 - BSD License
- Zstandard - BSD License
- NumPy - BSD License
- WebSockets - BSD License
- Flask - BSD License
- PySide6 - LGPL License

---

**GuiXi (龟息)** - Breathe slowly, transmit efficiently.
