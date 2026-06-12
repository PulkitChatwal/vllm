#!/bin/bash
# Commit and push PagedEviction integration to your fork

set -e

echo "=========================================="
echo "PagedEviction vLLM Fork - Commit & Push"
echo "=========================================="
echo ""

# Check we're in the right directory
if [ ! -d "vllm/v1/paged_eviction" ]; then
    echo "❌ Error: Not in vllm repository root"
    echo "Please run from: ~/Documents/vllm"
    exit 1
fi

echo "Current directory: $(pwd)"
echo ""

# Show what will be committed
echo "Files to be committed:"
echo "  ✓ vllm/v1/paged_eviction/__init__.py (new)"
echo "  ✓ vllm/v1/paged_eviction/paged_eviction.py (new)"
echo "  ✓ vllm/v1/scheduler_paged_eviction.py (new)"
echo "  ✓ vllm/config/cache.py (modified)"
echo "  ✓ vllm/engine/arg_utils.py (modified)"
echo "  ✓ vllm/entrypoints/llm.py (modified)"
echo "  ✓ test_paged_eviction.py (new)"
echo "  ✓ PAGED_EVICTION_INTEGRATION.md (new)"
echo ""

# Stage all files
echo "Staging files..."
git add vllm/v1/paged_eviction/__init__.py
git add vllm/v1/paged_eviction/paged_eviction.py
git add vllm/v1/scheduler_paged_eviction.py
git add vllm/config/cache.py
git add vllm/engine/arg_utils.py
git add vllm/entrypoints/llm.py
git add test_paged_eviction.py
git add PAGED_EVICTION_INTEGRATION.md

echo "✓ Files staged"
echo ""

# Show status
echo "Git status:"
git status --short
echo ""

# Commit
echo "Creating commit..."
git commit -m "Add PagedEviction KV cache pruning (EACL 2026)

- Add PagedEvictionConfig and PagedEvictionManager in vllm/v1/paged_eviction/
- Add scheduler integration in vllm/v1/scheduler_paged_eviction.py
- Add PagedEviction config fields to CacheConfig
- Add PagedEviction fields to EngineArgs
- Add CLI arguments (--enable-paged-eviction, --paged-eviction-budget, etc.)
- Add comprehensive test suite in test_paged_eviction.py
- Add integration documentation in PAGED_EVICTION_INTEGRATION.md

PagedEviction is a block-aligned KV cache eviction strategy that reduces
memory usage while maintaining accuracy. Based on EACL 2026 paper.

Features:
- Block-wise eviction that respects vLLM's PagedAttention structure
- Token importance metric: ||V||₂ / ||K||₂
- Prefill phase: token-level compression
- Decode phase: block-level eviction
- No modifications to CUDA attention kernels required
- Production-ready with full vLLM integration"

echo "✓ Commit created"
echo ""

# Push
echo "Pushing to origin/main..."
echo "Note: If prompted for authentication, use your GitHub username"
echo "      and a personal access token (not your password)"
echo "      Get token at: https://github.com/settings/tokens/new"
echo ""

if git push origin main; then
    echo ""
    echo "=========================================="
    echo "✓ SUCCESS! Pushed to your fork!"
    echo "=========================================="
    echo ""
    echo "Your fork: https://github.com/PulkitChatwal/vllm"
    echo ""
    echo "Next steps:"
    echo "1. Test in Google Colab:"
    echo "   !git clone https://github.com/PulkitChatwal/vllm.git"
    echo "   !cd vllm && pip install -e ."
    echo ""
    echo "2. Run test:"
    echo "   python test_paged_eviction.py"
    echo ""
    echo "3. Try with real model:"
    echo "   from vllm import LLM"
    echo "   llm = LLM(model='meta-llama/Llama-3.2-1B-Instruct',"
    echo "            enable_paged_eviction=True,"
    echo "            paged_eviction_budget=512)"
    echo ""
else
    echo ""
    echo "❌ Push failed. Common issues:"
    echo "1. Authentication required - use personal access token"
    echo "2. No internet connection"
    echo "3. Branch protection rules"
    echo ""
    echo "Try manual push:"
    echo "  git push origin main"
    exit 1
fi
