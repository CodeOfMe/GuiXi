"""
GuiXi (龟息) - Bandwidth-Efficient LLM Inference Framework

A framework for reducing bandwidth in LLM training and inference through
intelligent compression, semantic caching, delta synchronization, and
protocol optimization.
"""

from .__version__ import __version__
from .api import ToolResult, api_infer, api_batch_infer, api_cache_stats
from .tools import TOOLS, dispatch, list_tool_names
from .core import GuiXi, GuiXiClient, GuiXiServer
from .compress import CompressionEngine, CompressionMode
from .cache import SemanticCache, CachePolicy
from .protocol import BinaryProtocol, MessageType
from .sync import DeltaSync, DeltaUpdate

__all__ = [
    "__version__",
    "ToolResult",
    "api_infer",
    "api_batch_infer",
    "api_cache_stats",
    "TOOLS",
    "dispatch",
    "list_tool_names",
    "GuiXi",
    "GuiXiClient",
    "GuiXiServer",
    "CompressionEngine",
    "CompressionMode",
    "SemanticCache",
    "CachePolicy",
    "BinaryProtocol",
    "MessageType",
    "DeltaSync",
    "DeltaUpdate",
]
