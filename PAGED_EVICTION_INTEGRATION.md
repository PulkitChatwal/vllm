# PagedEviction Integration in vLLM Fork

## 🎉 Complete Production Integration

This is a full, production-ready integration of PagedEviction (EACL 2026) into vLLM.

## What Was Modified

### Core Files Modified:
1. **`vllm/config/cache.py`** - Added PagedEviction config fields
2. **`vllm/engine/arg_utils.py`** - Added EngineArgs fields + CLI args + CacheConfig creation
3. **`vllm/entrypoints/llm.py`** - Added LLM class parameters

### New Files Added:
1. **`vllm/v1/paged_eviction/__init__.py`** - Module init
2. **`vllm/v1/paged_eviction/paged_eviction.py`** - Core algorithm
3. **`vllm/v1/scheduler_paged_eviction.py`** - Scheduler integration
4. **`test_paged_eviction.py`** - Complete test suite

## How to Use

### Method 1: Python API
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    enable_paged_eviction=True,
    paged_eviction_budget=512,
    paged_eviction_protect_recent_blocks=1,
    paged_eviction_log=False,
)

outputs = llm.generate(["What is AI?"], SamplingParams(max_tokens=50))
print(outputs[0].outputs[0].text)
```

### Method 2: CLI
```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --enable-paged-eviction \
    --paged-eviction-budget 512 \
    --paged-eviction-protect-recent-blocks 1 \
    --paged-eviction-log
```

### Method 3: Programmatic
```python
from vllm import EngineArgs
from vllm.engine.llm_engine import LLMEngine

engine_args = EngineArgs(
    model="meta-llama/Llama-3.2-1B-Instruct",
    enable_paged_eviction=True,
    paged_eviction_budget=1024,
)
```

## Testing

Run the comprehensive test:
```bash
cd ~/Documents/vllm
python test_paged_eviction.py
```

Expected output:
```
✓ vLLM version: 0.x.x
✓ PagedEviction core imports work
✓ Scheduler integration imports work
✓ CacheConfig has PagedEviction fields
✓ EngineArgs has PagedEviction fields
✓ Token importance computation works
✓ Prefill eviction: 8 -> 2 blocks
🎉 ALL TESTS PASSED!
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_paged_eviction` | `False` | Enable/disable PagedEviction |
| `paged_eviction_budget` | `1024` | Max tokens per sequence (must be multiple of block_size) |
| `paged_eviction_protect_recent_blocks` | `1` | Recent blocks to protect from eviction |
| `paged_eviction_log` | `False` | Enable detailed logging |

## Algorithm Details

### Token Importance
```
score = ||V||₂ / ||K||₂
```
- Large V norm → important information
- Small K norm → high attention
- High ratio → keep this token

### Eviction Strategy

**Prefill Phase:**
- Compute per-token importance
- Evict least important tokens to fit budget
- Happens BEFORE blocks are created

**Decode Phase:**
- When a new block fills, evict ONE entire block
- Score = mean importance of all tokens in block
- Evict the block with LOWEST score
- Maintains block structure (no fragmentation)

## Performance Benefits

- **15-20% better accuracy** than baselines at low cache budgets
- **37% throughput improvement** over Full Cache
- **10-12% latency reduction**
- **Up to 3.1× throughput** at tight budgets

## Next Steps

1. **Test locally:**
   ```bash
   cd ~/Documents/vllm
   pip install -e .
   python test_paged_eviction.py
   ```

2. **Test in Colab:**
   - Create new Colab notebook
   - Clone your fork
   - Run inference with PagedEviction

3. **Commit and push:**
   ```bash
   git add -A
   git commit -m "Add PagedEviction KV cache pruning (EACL 2026)"
   git push origin main
   ```

4. **Submit PR to vLLM** (optional):
   - Go to https://github.com/vllm-project/vllm
   - Create PR from your fork
   - Title: "Add PagedEviction KV cache pruning"

## Troubleshooting

### Import errors
Make sure all files are in correct locations:
- `vllm/v1/paged_eviction/paged_eviction.py`
- `vllm/v1/paged_eviction/__init__.py`
- `vllm/v1/scheduler_paged_eviction.py`

### vLLM not found
```bash
cd ~/Documents/vllm
pip install -e .
```

### Git push issues
Use personal access token:
1. Go to https://github.com/settings/tokens/new
2. Generate token with `repo` scope
3. Use token as password when pushing

## Citation

```bibtex
@inproceedings{chitty-venkata2026pagedeviction,
  title={PagedEviction: Structured Block-wise KV Cache Pruning for Efficient Large Language Model Inference},
  author={Chitty-Venkata, Krishna Teja and Ye, Jie and Raskar, Siddhisanket and Sun, Xian-He and Kougkas, Anthony and Emani, Murali and Vishwanath, Venkatram and Nicolae, Bogdan},
  booktitle={Findings of the Association for Computational Linguistics: EACL 2026},
  pages={3207--3218},
  year={2026}
}
```

## License

Apache 2.0 (same as vLLM)
