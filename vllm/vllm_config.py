"""
vLLM Configuration with PagedEviction Support

This modifies vLLM's configuration to support PagedEviction.
Add this to vLLM's existing config system.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PagedEvictionArgs:
    """Arguments for PagedEviction."""

    enabled: bool = False
    cache_budget: int = 1024
    block_size: int = 16
    protect_recent_blocks: int = 1
    enable_prefill_eviction: bool = True
    enable_decode_eviction: bool = True
    log_evictions: bool = False

    def __post_init__(self):
        if self.cache_budget <= 0:
            raise ValueError("cache_budget must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")


def add_paged_eviction_args(parser):
    """Add PagedEviction arguments to vLLM's argument parser."""
    group = parser.add_argument_group("PagedEviction options")
    group.add_argument(
        "--enable-paged-eviction",
        action="store_true",
        default=False,
        help="Enable PagedEviction KV cache pruning",
    )
    group.add_argument(
        "--paged-eviction-cache-budget",
        type=int,
        default=1024,
        help="Maximum KV cache tokens to retain per sequence",
    )
    group.add_argument(
        "--paged-eviction-block-size",
        type=int,
        default=16,
        help="Block size for paged eviction",
    )
    group.add_argument(
        "--paged-eviction-protect-recent-blocks",
        type=int,
        default=1,
        help="Number of recent blocks to protect from eviction",
    )
    group.add_argument(
        "--paged-eviction-log",
        action="store_true",
        default=False,
        help="Log eviction events",
    )
    return parser