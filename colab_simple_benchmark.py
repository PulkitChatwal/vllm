#!/usr/bin/env python3
"""
Simple Colab Benchmark - PagedEviction vs Standard vLLM
Copy this entire script to a Colab cell
"""

import torch
import time
import gc
import json
from datetime import datetime
from vllm import LLM, SamplingParams

print("=" * 60)
print("PagedEviction vs Standard vLLM Benchmark")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test prompts (long contexts to stress KV cache)
test_prompts = [
    """The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain.

The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and they were given millions of dollars to make this vision come true.

Unfortunately, it soon became obvious that they had grossly underestimated the difficulty of the problem. Progress stalled and in 1973, in response to the criticism of Sir James Lighthill and ongoing pressure from the US Congress to fund more productive projects, both the U.S. and British governments cut off exploratory research in AI. The next few years would later be called an "AI winter", a period when funding for AI projects was hard to find.

In the early 1980s, AI research was revived by the commercial success of expert systems, a form of AI program that simulated the knowledge and analytical skills of human experts. By 1985, the market for AI had reached over a billion dollars. At the same time, Japan's fifth generation computer project inspired the U.S and British governments to restore funding for academic AI research. However, beginning with the collapse of the Lisp Machine market in 1987, AI once again fell into disrepute, and a second, longer-lasting winter began.

Interest in AI as a field of research revived in the 1990s and early 2000s with the success of deep learning methods. These methods proved to be highly effective for many of the problems that had been difficult to solve with earlier methods. The 2010s saw a massive surge in AI research and applications, with deep learning being used for everything from image recognition to natural language processing to game playing.

The most recent advances in AI have come from large language models (LLMs) like GPT-3, GPT-4, and others. These models, trained on massive amounts of text data, have shown remarkable capabilities in understanding and generating human language. They can write essays, answer questions, translate languages, and even generate code. However, they also come with significant challenges, including high computational costs, large memory requirements, and potential for generating misleading or harmful content.

One of the key challenges with LLMs is the KV cache, which stores the key and value states from the attention mechanism. As the sequence length increases, this cache grows linearly, quickly becoming a major memory bottleneck. PagedEviction is a novel approach to managing this cache, using block-wise eviction to maintain performance while reducing memory usage.""" * 3,

    """Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations. The devices that perform quantum computations are known as quantum computers. Although current quantum computers are too small to outperform usual (classical) computers for practical applications, they are believed to be capable of solving certain computational problems, such as integer factorization (which underlies RSA encryption), substantially faster than classical computers.

The study of quantum computing is a subfield of quantum information science. Quantum computing began in the early 1980s when physicist Paul Benioff proposed a quantum mechanical model of the Turing machine. Richard Feynman and Yuri Manin later suggested that a quantum computer had the potential to simulate things that a classical computer could not feasibly do. In 1994, Peter Shor developed a quantum algorithm for factoring integers that had the potential to decrypt RSA-encrypted communications. Although Shor's algorithm is polynomial in the number of bits, the number of qubits required is much larger than the number of bits in the number to be factored.

Despite ongoing experimental progress since the late 1990s, most researchers believe that "fault-tolerant quantum computing [is] still a rather distant dream". As of 2023, classical computers outperform quantum computers for all real-world applications, and there is no known way to efficiently simulate quantum computers using classical computers.

Recent advances in quantum computing hardware have been significant. IBM, Google, and other companies have demonstrated quantum computers with over 100 qubits. These systems, while still far from the millions of qubits needed for practical applications, represent important milestones in the field. The development of error correction techniques, better qubit designs, and more sophisticated control systems are all areas of active research.

One of the major challenges in quantum computing is decoherence, which is the loss of quantum coherence or the loss of information due to the environment. Quantum systems are extremely sensitive to their environment, and any interaction with the outside world can cause the system to lose its quantum properties. This makes it very difficult to maintain the delicate quantum states needed for computation.

To address this challenge, researchers are developing various approaches, including topological qubits, which are more resistant to decoherence, and quantum error correction codes, which can detect and correct errors without destroying the quantum information. These advances, combined with improvements in fabrication and control systems, are gradually bringing practical quantum computing closer to reality.""" * 3,

    """Climate change is one of the most pressing issues facing humanity today. The Earth's climate has changed throughout history. Most of these changes can be attributed to very small variations in Earth's orbit that change the amount of solar energy our planet receives. However, since the industrial revolution, human activities have been the main driver of climate change.

The primary cause of current climate change is the burning of fossil fuels, which releases carbon dioxide and other greenhouse gases into the atmosphere. These gases trap heat from the sun, causing the Earth's average temperature to rise. The consequences of this warming are far-reaching and include more frequent and severe weather events, rising sea levels, and disruptions to ecosystems and agriculture.

The Paris Agreement, signed in 2016, represents a global effort to combat climate change. Its goal is to limit global warming to well below 2 degrees Celsius, preferably to 1.5 degrees, compared to pre-industrial levels. To achieve this, countries have committed to reducing their greenhouse gas emissions and transitioning to clean energy sources.

However, progress has been slow, and many scientists believe that current commitments are insufficient to meet the Paris Agreement goals. The Intergovernmental Panel on Climate Change (IPCC) has warned that limiting warming to 1.5 degrees will require unprecedented changes in all aspects of society, including energy production, transportation, industry, and agriculture.

Renewable energy sources, such as solar and wind power, have become increasingly cost-competitive with fossil fuels in recent years. This has led to rapid growth in renewable energy capacity around the world. However, the transition to a clean energy economy is complex and requires significant investments in infrastructure, technology, and policy.

In addition to reducing emissions, many scientists argue that we will also need to remove carbon dioxide from the atmosphere to limit warming to 1.5 degrees. This could be achieved through natural solutions, such as reforestation and soil carbon sequestration, or through technological solutions, such as direct air capture and carbon capture and storage.

The impacts of climate change are already being felt around the world, from more frequent heat waves and droughts to more intense hurricanes and floods. Vulnerable communities, particularly in developing countries, are disproportionately affected by these impacts. Addressing climate change will require global cooperation, technological innovation, and significant changes in how we produce and consume energy.""" * 3,

    """The human brain is the central organ of the human nervous system, and with the spinal cord makes up the central nervous system. The brain consists of the cerebrum, the brainstem, and the cerebellum. It controls most of the activities of the body, processing, integrating, and coordinating the information it receives from the sense organs, and making decisions as to the instructions sent to the rest of the body.

The brain is contained in, and protected by, the skull bones of the head. The cerebrum is the largest part of the brain and is composed of two hemispheres, the left and right, which are connected by a bundle of nerve fibers called the corpus callosum. The cerebrum is divided into four lobes: the frontal lobe, parietal lobe, temporal lobe, and occipital lobe. Each lobe has different functions, although they work together in complex ways.

The frontal lobe is associated with executive functions, including self-control, planning, reasoning, and abstract thought. The parietal lobe processes sensory information such as touch, temperature, pain, and proprioception. The temporal lobe is involved in processing auditory information and is also important for memory and emotion. The occipital lobe is primarily responsible for vision.

The brainstem connects the cerebrum to the spinal cord and controls many basic life functions, including breathing, heart rate, and blood pressure. The cerebellum is located at the back of the brain and is involved in coordination, balance, and fine motor control.

Neurons are the basic functional units of the brain, and there are approximately 86 billion neurons in the human brain. Each neuron can form connections with thousands of other neurons, creating a vast network of communication. These connections, called synapses, are the basis for all brain functions, from simple reflexes to complex thought and emotion.

Neuroscience is the scientific study of the nervous system, including the brain. It is a multidisciplinary field that combines biology, chemistry, physics, psychology, and computer science. Advances in neuroscience have led to a better understanding of how the brain works and have implications for treating neurological and psychiatric disorders.

Recent advances in brain imaging technologies, such as functional magnetic resonance imaging (fMRI) and optogenetics, have allowed scientists to observe the brain in action and manipulate specific neurons. These tools are providing unprecedented insights into how the brain works and are opening new avenues for treating brain disorders.""" * 3,

    """The history of the internet is a fascinating journey that began in the 1960s with the development of ARPANET, a project funded by the U.S. Department of Defense. ARPANET was designed to enable communication between different computers at research institutions, and it used a novel technology called packet switching to send data efficiently across the network.

The development of ARPANET led to the creation of the Transmission Control Protocol and Internet Protocol (TCP/IP) in the 1970s, which became the foundation for the modern internet. TCP/IP allowed different networks to communicate with each other, creating a network of networks. This was a significant breakthrough that enabled the internet to grow and evolve.

In the 1980s, the National Science Foundation (NSF) created NSFNET, a high-speed network that connected research institutions across the United States. NSFNET was much faster than ARPANET and helped to popularize the internet among researchers and academics. The NSF also allowed commercial use of the network, which led to the growth of internet service providers (ISPs) and the commercialization of the internet.

The 1990s saw the explosive growth of the internet, with the development of the World Wide Web by Tim Berners-Lee in 1989-1991. The Web made it easy for non-technical users to access and share information on the internet, and it led to the creation of the first web browsers, such as Mosaic and Netscape Navigator. This made the internet accessible to a much wider audience and transformed many aspects of society.

The 2000s saw the rise of social media, e-commerce, and mobile internet. Companies like Google, Facebook, and Amazon became household names, and the internet became an integral part of daily life for billions of people around the world. The development of smartphones and mobile networks made it possible to access the internet from anywhere, further accelerating its growth and impact.

Today, the internet is a global network that connects billions of devices and people. It has revolutionized virtually every aspect of modern life, from how we communicate and access information to how we shop, work, and entertain ourselves. However, it also raises important questions about privacy, security, and the digital divide.

The future of the internet is likely to be shaped by emerging technologies such as artificial intelligence, the Internet of Things (IoT), and quantum computing. These technologies promise to bring new capabilities and applications, but they also raise new challenges and concerns that will need to be addressed as the internet continues to evolve.""" * 3,
]

print(f"\n✓ Created {len(test_prompts)} test prompts")
for i, prompt in enumerate(test_prompts):
    print(f"  Prompt {i+1}: {len(prompt)} chars, ~{len(prompt.split())} words")


def benchmark_model(llm, prompts, max_tokens=256, name="Model"):
    """Benchmark a model with given prompts"""
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
    )

    # Warmup
    print(f"\nWarming up {name}...")
    _ = llm.generate(["Hello"], SamplingParams(max_tokens=10))

    # Reset GPU stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    # Benchmark
    print(f"Running benchmark for {name}...")
    start_time = time.time()

    outputs = llm.generate(prompts, sampling_params)

    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    throughput = total_tokens / total_time
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    avg_time_per_prompt = total_time / len(prompts)

    metrics = {
        "name": name,
        "total_time": total_time,
        "total_tokens": total_tokens,
        "throughput_tokens_per_sec": throughput,
        "peak_memory_gb": peak_memory,
        "avg_time_per_prompt": avg_time_per_prompt,
        "num_prompts": len(prompts),
        "max_tokens": max_tokens,
    }

    return metrics, outputs


# ============================================
# BENCHMARK 1: Standard vLLM
# ============================================
print("\n" + "=" * 60)
print("BENCHMARK 1: Standard vLLM (No PagedEviction)")
print("=" * 60)

llm_standard = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    gpu_memory_utilization=0.9,
    max_model_len=8192,
)

standard_metrics, standard_outputs = benchmark_model(
    llm_standard,
    test_prompts,
    max_tokens=256,
    name="Standard vLLM"
)

print(f"\n✓ Standard vLLM Results:")
print(f"  Total time: {standard_metrics['total_time']:.2f}s")
print(f"  Total tokens: {standard_metrics['total_tokens']}")
print(f"  Throughput: {standard_metrics['throughput_tokens_per_sec']:.2f} tokens/sec")
print(f"  Peak memory: {standard_metrics['peak_memory_gb']:.2f} GB")
print(f"  Avg time/prompt: {standard_metrics['avg_time_per_prompt']:.2f}s")

# Free memory
del llm_standard
gc.collect()
torch.cuda.empty_cache()


# ============================================
# BENCHMARK 2: PagedEviction vLLM
# ============================================
print("\n" + "=" * 60)
print("BENCHMARK 2: vLLM with PagedEviction")
print("=" * 60)

llm_paged = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    enable_paged_eviction=True,  # ← PagedEviction enabled!
    paged_eviction_budget=1024,
)

paged_metrics, paged_outputs = benchmark_model(
    llm_paged,
    test_prompts,
    max_tokens=256,
    name="PagedEviction vLLM"
)

print(f"\n✓ PagedEviction Results:")
print(f"  Total time: {paged_metrics['total_time']:.2f}s")
print(f"  Total tokens: {paged_metrics['total_tokens']}")
print(f"  Throughput: {paged_metrics['throughput_tokens_per_sec']:.2f} tokens/sec")
print(f"  Peak memory: {paged_metrics['peak_memory_gb']:.2f} GB")
print(f"  Avg time/prompt: {paged_metrics['avg_time_per_prompt']:.2f}s")


# ============================================
# COMPARISON
# ============================================
print("\n" + "=" * 60)
print("COMPARISON: PagedEviction vs Standard vLLM")
print("=" * 60)

# Calculate improvements
time_speedup = standard_metrics['total_time'] / paged_metrics['total_time']
throughput_speedup = paged_metrics['throughput_tokens_per_sec'] / standard_metrics['throughput_tokens_per_sec']
memory_saved = standard_metrics['peak_memory_gb'] - paged_metrics['peak_memory_gb']
memory_saved_pct = (memory_saved / standard_metrics['peak_memory_gb']) * 100

print(f"\n📊 Performance Comparison:")
print(f"  Throughput improvement: {throughput_speedup:.2f}x")
print(f"  Time speedup: {time_speedup:.2f}x")
print(f"  Memory saved: {memory_saved:.2f} GB ({memory_saved_pct:.1f}%)")

print(f"\n⏱️  Time Comparison:")
print(f"  Standard vLLM: {standard_metrics['total_time']:.2f}s")
print(f"  PagedEviction: {paged_metrics['total_time']:.2f}s")
print(f"  Faster by: {standard_metrics['total_time'] - paged_metrics['total_time']:.2f}s")

print(f"\n🚀 Throughput Comparison:")
print(f"  Standard vLLM: {standard_metrics['throughput_tokens_per_sec']:.2f} tokens/sec")
print(f"  PagedEviction: {paged_metrics['throughput_tokens_per_sec']:.2f} tokens/sec")
print(f"  Gain: +{paged_metrics['throughput_tokens_per_sec'] - standard_metrics['throughput_tokens_per_sec']:.2f} tokens/sec")

print(f"\n💾 Memory Comparison:")
print(f"  Standard vLLM: {standard_metrics['peak_memory_gb']:.2f} GB")
print(f"  PagedEviction: {paged_metrics['peak_memory_gb']:.2f} GB")
print(f"  Saved: {memory_saved:.2f} GB ({memory_saved_pct:.1f}%)")


# ============================================
# VISUALIZATION
# ============================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Throughput
axes[0].bar(['Standard', 'PagedEviction'],
            [standard_metrics['throughput_tokens_per_sec'],
             paged_metrics['throughput_tokens_per_sec']],
            color=['#ff7f0e', '#2ca02c'])
axes[0].set_ylabel('Tokens/sec')
axes[0].set_title('Throughput Comparison')
axes[0].grid(True, alpha=0.3)

# Time
axes[1].bar(['Standard', 'PagedEviction'],
            [standard_metrics['total_time'], paged_metrics['total_time']],
            color=['#ff7f0e', '#2ca02c'])
axes[1].set_ylabel('Time (seconds)')
axes[1].set_title('Total Time Comparison')
axes[1].grid(True, alpha=0.3)

# Memory
axes[2].bar(['Standard', 'PagedEviction'],
            [standard_metrics['peak_memory_gb'], paged_metrics['peak_memory_gb']],
            color=['#ff7f0e', '#2ca02c'])
axes[2].set_ylabel('Memory (GB)')
axes[2].set_title('Peak Memory Comparison')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('benchmark_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n✓ Plot saved as benchmark_comparison.png")


# ============================================
# SAVE RESULTS
# ============================================
results = {
    "timestamp": datetime.now().isoformat(),
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "num_prompts": len(test_prompts),
    "max_tokens": 256,
    "standard_vllm": standard_metrics,
    "paged_eviction": paged_metrics,
    "improvements": {
        "throughput_speedup": throughput_speedup,
        "time_speedup": time_speedup,
        "memory_saved_gb": memory_saved,
        "memory_saved_pct": memory_saved_pct,
    },
}

with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to benchmark_results.json")
print("\n" + "=" * 60)
print("🎉 BENCHMARK COMPLETE!")
print("=" * 60)
print(f"\nSummary:")
print(f"  • Throughput: {throughput_speedup:.2f}x faster")
print(f"  • Time: {time_speedup:.2f}x speedup")
print(f"  • Memory: {memory_saved_pct:.1f}% saved")
print("\n" + "=" * 60)
