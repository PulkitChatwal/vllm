"""
PagedEviction: Block-Aligned KV Cache Pruning for vLLM

This module is fully integrated into vLLM's V1 engine.
Production-ready implementation of the EACL 2026 paper.
"""

# Expose the public API
from vllm.v1.paged_eviction.paged_eviction import (
    PagedEvictionConfig,
    PagedEvictionManager,
    PagedEvictionStats,
    PagedEvictionPreset,
)

__all__ = [
    "PagedEvictionConfig",
    "PagedEvictionManager",
    "PagedEvictionStats",
    "PagedEvictionPreset",
]
