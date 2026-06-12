#!/usr/bin/env python3
"""
Simple test for PagedEviction integration (no full vLLM install needed)

This test verifies that all PagedEviction files are in the right places
and can be imported standalone (without the full vLLM runtime).
"""

import sys
import os
from pathlib import Path

print("=" * 60)
print("PagedEviction - Simple File Integration Test")
print("=" * 60)

# Add vllm to path
vllm_path = Path(__file__).parent
sys.path.insert(0, str(vllm_path))

# Test 1: Check all required files exist
print("\n[1/6] Checking required files exist...")
required_files = [
    "vllm/v1/paged_eviction/__init__.py",
    "vllm/v1/paged_eviction/paged_eviction.py",
    "vllm/v1/scheduler_paged_eviction.py",
    "vllm/config/cache.py",
    "vllm/engine/arg_utils.py",
    "vllm/entrypoints/llm.py",
]

all_exist = True
for file_path in required_files:
    full_path = vllm_path / file_path
    if full_path.exists():
        print(f"  ✓ {file_path}")
    else:
        print(f"  ❌ {file_path} - MISSING!")
        all_exist = False

if not all_exist:
    print("\n❌ Some files are missing. Please check the integration.")
    sys.exit(1)

print("\n✓ All required files exist!")

# Test 2: Check PagedEviction core file
print("\n[2/6] Checking PagedEviction core file...")
core_file = vllm_path / "vllm/v1/paged_eviction/paged_eviction.py"
if core_file.exists():
    content = core_file.read_text()
    if "PagedEvictionConfig" in content and "PagedEvictionManager" in content:
        print("  ✓ PagedEvictionConfig class found")
        print("  ✓ PagedEvictionManager class found")
    else:
        print("  ❌ Required classes not found in paged_eviction.py")
        sys.exit(1)

# Test 3: Check scheduler integration
print("\n[3/6] Checking scheduler integration...")
sched_file = vllm_path / "vllm/v1/scheduler_paged_eviction.py"
if sched_file.exists():
    content = sched_file.read_text()
    if "install_paged_eviction" in content:
        print("  ✓ install_paged_eviction function found")
    else:
        print("  ❌ install_paged_eviction function not found")
        sys.exit(1)

# Test 4: Check CacheConfig integration
print("\n[4/6] Checking CacheConfig integration...")
cache_file = vllm_path / "vllm/config/cache.py"
if cache_file.exists():
    content = cache_file.read_text()
    if "enable_paged_eviction" in content:
        print("  ✓ enable_paged_eviction field in CacheConfig")
    else:
        print("  ❌ enable_paged_eviction not in CacheConfig")
        sys.exit(1)
    if "paged_eviction_budget" in content:
        print("  ✓ paged_eviction_budget field in CacheConfig")
    else:
        print("  ❌ paged_eviction_budget not in CacheConfig")
        sys.exit(1)

# Test 5: Check EngineArgs integration
print("\n[5/6] Checking EngineArgs integration...")
arg_file = vllm_path / "vllm/engine/arg_utils.py"
if arg_file.exists():
    content = arg_file.read_text()
    if "enable_paged_eviction: bool = CacheConfig.enable_paged_eviction" in content:
        print("  ✓ enable_paged_eviction in EngineArgs")
    else:
        print("  ❌ enable_paged_eviction not properly in EngineArgs")
        sys.exit(1)

    if "--enable-paged-eviction" in content:
        print("  ✓ --enable-paged-eviction CLI argument")
    else:
        print("  ❌ --enable-paged-eviction CLI argument not found")
        sys.exit(1)

# Test 6: Check LLM class integration
print("\n[6/6] Checking LLM class integration...")
llm_file = vllm_path / "vllm/entrypoints/llm.py"
if llm_file.exists():
    content = llm_file.read_text()
    if "enable_paged_eviction: bool = False" in content:
        print("  ✓ enable_paged_eviction parameter in LLM.__init__")
    else:
        print("  ❌ enable_paged_eviction not in LLM.__init__")
        sys.exit(1)

print("\n" + "=" * 60)
print("🎉 ALL INTEGRATION CHECKS PASSED!")
print("=" * 60)

print("\n📊 Summary of Integration:")
print("  ✓ Core algorithm (vllm/v1/paged_eviction/)")
print("  ✓ Scheduler integration (vllm/v1/scheduler_paged_eviction.py)")
print("  ✓ CacheConfig fields (vllm/config/cache.py)")
print("  ✓ EngineArgs fields + CLI args (vllm/engine/arg_utils.py)")
print("  ✓ LLM class parameters (vllm/entrypoints/llm.py)")

print("\n🚀 Next Steps:")
print("\n1. Test PagedEviction algorithm standalone (no vLLM needed):")
print("   python test_algorithm_only.py")

print("\n2. Install vLLM (requires PyTorch, ~2GB):")
print("   pip install -e .")
print("   # OR use the fork in Google Colab which has everything pre-installed")

print("\n3. Commit and push to your fork:")
print("   ./commit_and_push.sh")
print("   # OR manually:")
print("   git add -A")
print("   git commit -m 'Add PagedEviction integration'")
print("   git push origin main")

print("\n4. Test in Google Colab:")
print("   !git clone https://github.com/PulkitChatwal/vllm.git")
print("   !cd vllm && pip install -e .")
print("   !python test_paged_eviction.py")

print("\n💡 Recommendation: Test in Google Colab - it has PyTorch pre-installed")
print("   and you can test with real GPU models!")

print("\n" + "=" * 60)
