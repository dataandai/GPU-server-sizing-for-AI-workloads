# Capacity Planner — Parameter Reference Guide

## 📋 BASICS AND MODEL SOURCE

### Scenario name
The name of the current simulation run, which appears in the output file and the report. Helps you track different configurations and runs across sessions.

> **Example:** `llama3-70b-hungarian-batch-v2` — makes it immediately clear which model, language, and workload type this run refers to.

### Model source
There are three ways to select a model: from the Hugging Face catalogue (dynamic download), a built-in base model, or manually specified model parameters (inline mode). Inline mode is recommended when you need precise KV cache calculations or a custom attention architecture.

> **Example:** Choose `inline` if you are configuring a custom-tuned MoE model whose config is not yet published on Hugging Face.

### Target language
The linguistic profile of the simulated workload (Hungarian, English, German). The Hungarian setting applies a conservative token over-estimation, because the same meaning typically requires more tokens in Hungarian than in English.

> **Example:** Setting `Hungarian` increases estimated token counts by ~20–30% compared to `English` for equivalent text — important for realistic memory planning.

---

## 🤖 HUGGING FACE MODEL LOADING

### Hugging Face model_id
The name of the model available on Hugging Face Hub, e.g. `meta-llama/Llama-3.1-70B-Instruct`. The system downloads the model configuration and parameters from the network.

> **Example:** `mistralai/Mistral-7B-Instruct-v0.3` — the system fetches its layer count, hidden size, and attention config automatically.

### Revision / tag
Optional commit hash or branch version for the model (e.g. `main` or a specific commit ID). If left empty, the default branch is used.

> **Example:** `a1b2c3d` pins the simulation to an exact snapshot, ensuring reproducibility even if the model is updated on Hub later.

### Local cache only?
If enabled, the Hugging Face API only searches the local cache and does not download from the internet. Useful for offline mode or faster testing.

> **Example:** Enable this in an air-gapped data-center environment where outbound internet access is restricted.

### Note
A free-text comment for the given Hugging Face configuration, helping document the purpose of the run.

> **Example:** `"Baseline run before quantization — FP16 full weights, no AWQ"`.

---

## 📊 INLINE MANUAL MODEL CONFIG

### Name
The model identifier used by the simulation engine in the YAML and output files. Can be any arbitrary name.

> **Example:** `custom-llm`, `deepseek-r1-671b-moe`, `internal-finetuned-v3`.

### Total parameters (B)
The total number of model parameters in billions (e.g. 70B). Useful for MoE models where the total parameter count is larger than the actively used portion.

> **Example:** A Mixtral 8×22B model has ~141B total parameters, but only ~39B are active per token.

### Active parameters (B)
The parameters actually active during the forward pass, in billions. For MoE models this is smaller than the total parameter count.

> **Example:** For Mixtral 8×22B with 2 active experts, set `Active parameters = 39`.

### Layer count
The number of transformer layers in the model. A higher number generally means greater KV cache and compute requirements.

> **Example:** Llama 3.1 70B has 80 layers; doubling the layer count roughly doubles KV cache memory at the same batch size.

### Hidden size
The size of the hidden dimension (e.g. 8192). Critical for KV cache calculation: a larger value means more memory is required.

> **Example:** `hidden_size = 8192` for a 70B model vs. `4096` for a 7B model — the former needs ~4× more memory per token in the KV cache.

### Head dim
The dimension of a single attention head (typically 128). Together with the number of attention heads, this determines the hidden size.

> **Example:** With `attention_heads = 64` and `head_dim = 128`, the hidden size is 64 × 128 = 8192.

### Attention heads
The total number of attention heads (e.g. 64). This value influences the dimensionality of the KV cache.

> **Example:** A model with 64 attention heads and FP16 KV cache uses twice as much KV memory per layer as a model with 32 heads.

### KV heads
Specifies how many heads have dedicated Key and Value sections. In GQA and MQA architectures this is much smaller than the number of attention heads.

> **Example:** Llama 3 70B uses GQA with `kv_heads = 8` vs. `attention_heads = 64`, reducing KV cache size by 8×.

### Intermediate size
The size of the FFN (feed-forward network) hidden dimension. A larger value means more parameters and more computation after the attention block.

> **Example:** `intermediate_size = 28672` for a 70B model; increasing it to 32768 adds ~10% more parameters to each FFN layer.

### MoE intermediate size
The per-expert FFN dimension in Mixture of Experts models. Leave empty or set to 0 for non-MoE models.

> **Example:** In DeepSeek-V2, each expert has `moe_intermediate_size = 1536` while the model has 160 experts total.

### Expert count
The total number of experts in an MoE model. Leave empty or set to 0 for standard dense models.

> **Example:** Mixtral 8×7B has `expert_count = 8`; Grok-1 has `expert_count = 8` with a much larger per-expert size.

### Experts / token
How many experts are activated simultaneously per token in MoE models. Leave empty or set to 0 for standard models.

> **Example:** Mixtral uses `experts_per_token = 2`; DeepSeek-V2 uses `experts_per_token = 6` out of 160 total experts.

### Vocab size
The size of the tokenizer vocabulary (e.g. 128256). This affects the embedding table size and the output logit dimension.

> **Example:** Llama 3 uses `vocab_size = 128256`; older Llama 2 used 32000 — the larger vocabulary increases the embedding layer's VRAM footprint.

### Max position embeddings
The maximum sequence length supported by the model, in tokens (e.g. 131072). This is the upper limit for KV cache sizing.

> **Example:** A model with `max_position_embeddings = 131072` can handle ~100K-token contexts, but the KV cache scales linearly with sequence length.

### MoE model?
Boolean flag (yes/no) indicating whether the model uses a Mixture of Experts architecture. This affects parameter calculations and memory estimation.

> **Example:** Set to `yes` for Mixtral, DeepSeek-V2, or Grok-1; leave `no` for dense models like Llama or Mistral.

### Attention architecture
The specific attention architecture type: Dense MHA, GQA, MQA, DeepSeek MLA, Hybrid Transformer-Mamba, Sliding-Window, Pure SSM/Mamba, or FFN-MoE. This is **critical** for KV cache multiplier calculation.

> **Example:** Selecting `DeepSeek MLA` triggers a ~67% KV cache memory saving compared to standard MHA, while `GQA` saves memory proportional to the ratio of KV heads to attention heads.

---

## 🖥️ HARDWARE, STACK AND DEPLOYMENT

### Vendor filter
Filter hardware by manufacturer (e.g. Dell, NVIDIA, AMD). Narrows down the available hardware options in the catalogue.

> **Example:** Set `Vendor = NVIDIA` to see only DGX and HGX systems and exclude AMD Instinct-based platforms.

### GPU count filter
The number of GPUs to filter by: 2, 4, 8, or all. The UI only shows hardware configurations that match the selection.

> **Example:** Select `8` to find servers suitable for TP=8 tensor parallelism with a 70B+ model.

### Hardware catalog ID
The specific hardware server selected from the catalogue (e.g. `nvidia_dgx_h200_8gpu_latest`). This includes GPU, CPU, memory, and NVLink topology.

> **Example:** `dell_xe9680_8xa100_80gb` selects a Dell PowerEdge XE9680 with 8× A100 80 GB GPUs.

### Software stack
The combination of serving engine and CUDA/cuDNN versions (e.g. `nvidia_vllm_cuda`, `nvidia_tensorrt_llm`). This affects overhead and performance.

> **Example:** `nvidia_vllm_cuda` uses vLLM's PagedAttention for efficient KV cache management; `nvidia_tensorrt_llm` may offer higher throughput for batch inference.

### Deployment profile
The operational setup: bare metal, Kubernetes, VM passthrough, vGPU time slicing, or MIG. This affects additional overhead and resource sharing.

> **Example:** `mig` (Multi-Instance GPU) allows a single H100 to be partitioned into isolated GPU slices, reducing per-tenant VRAM but adding ~5–10% overhead.

### Tensor parallel
The model sharding level (e.g. TP=2, TP=4, TP=8). Recommended values are derived from the selected hardware (e.g. a Dell node may suggest TP=2).

> **Example:** A 70B model in FP16 requires ~140 GB VRAM; with 8× 80 GB GPUs and `TP=8`, each GPU holds ~17.5 GB of weights.

---

## 💾 INFERENCE AND MEMORY SETTINGS

### Weight quantization
The bit-width of LLM weights: INT8, FP16, BF16, INT4, or NVFP4 (Blackwell only). Lower bit-width = less VRAM, but potentially lower quality.

> **Example:** A 70B model at FP16 needs ~140 GB; at INT4 it drops to ~35 GB — fitting on a single 4× 80 GB node instead of two nodes.

### KV cache precision
The bit-width of Key/Value caches: FP16, BF16, FP8, or INT8. FP8 uses less memory but may introduce minor accuracy loss.

> **Example:** Switching from FP16 to FP8 KV cache halves KV memory usage, allowing roughly 2× more concurrent tokens at the same batch size.

### Runtime overhead (%)
The extra cost of containerization, virtualization, and system processes, expressed as a percentage (e.g. 10%). This is deducted from total VRAM before planning.

> **Example:** A Kubernetes + containerd deployment typically adds 8–12% overhead; bare metal deployments can use as low as 3–5%.

### Max VRAM utilization (%)
The maximum percentage of available VRAM that can be used for the workload (e.g. 95%). The remainder acts as a buffer for the OS and edge cases.

> **Example:** With 80 GB VRAM and `max_utilization = 95%`, only 76 GB is available for model weights + KV cache.

### Safety headroom (%)
A safety margin added on top of the estimated memory requirement, to handle extreme cases (e.g. 5%).

> **Example:** With estimated usage of 70 GB and `safety_headroom = 5%`, the planner targets a maximum of 73.5 GB, rejecting configurations that exceed it.

---

## 📦 WORKLOAD TYPES — FIXED BATCH

### Prompt tokens
The average prompt length in tokens for a batch request (e.g. 1200). This affects the KV cache size at the time of the first token generated.

> **Example:** Document summarization jobs often have `prompt_tokens = 4000+`; simple classification prompts may be as short as `128`.

### Max output tokens
The maximum output sequence length in tokens (e.g. 500). This is the upper limit per user in the workload.

> **Example:** A batch translation job might use `max_output_tokens = 2000`; a sentiment classification job might only need `64`.

### Concurrent jobs
The number of batch jobs running simultaneously (e.g. 10). This directly affects the total KV cache size.

> **Example:** `concurrent_jobs = 50` with `prompt_tokens = 1000` and `max_output_tokens = 500` means the planner must reserve KV cache for 50 × 1500 = 75,000 active tokens.

---

## 🤖 WORKLOAD TYPES — AGENTIC

### Initial prompt tokens
The agent's initial system/context message in tokens (e.g. 1100). This remains in context for the entire duration of the agent run.

> **Example:** A coding agent with a large system prompt describing tools and guidelines may have `initial_prompt_tokens = 3000`.

### Concurrent agents
The number of simultaneously running agentic processes (e.g. 10). More agents = larger total KV cache demand.

> **Example:** Running 20 concurrent research agents each with a 2000-token context requires planning for 40,000 tokens of KV cache at minimum.

### Max turn
The maximum number of turns for the agent (e.g. 10). More turns = more tokens and a longer context window needed.

> **Example:** A simple Q&A agent may complete in `max_turn = 3`; a complex planning agent may need `max_turn = 20+`.

### Tool call probability (%)
The probability (in percent) that the agent calls a tool at a given step (e.g. 70%). Higher probability → more token consumption.

> **Example:** With `tool_call_probability = 80%` and `max_turn = 10`, the agent is expected to make ~8 tool calls per run on average.

### Tool result tokens
The average token count of a single tool response (e.g. 400). This is added to the context length.

> **Example:** A web search tool returning a snippet may produce `tool_result_tokens = 300`; a database query returning rows could produce `2000+`.

### Assistant response tokens
The average response length in tokens when the agent is not making a tool call (e.g. 250). This influences the length of each turn.

> **Example:** An agent that summarizes findings between tool calls may produce `assistant_response_tokens = 500` per turn.

### Stop probability / turn (%)
The probability (in percent) that the agent stops at a given turn (e.g. 15%). Higher → shorter average agent runs.

> **Example:** With `stop_probability = 20%` per turn, the expected number of turns is ~5 (geometric distribution mean = 1/0.2).

### Max tokens / run
The maximum number of tokens for a complete agent run (e.g. 32768). This caps the exponential growth of the context.

> **Example:** Setting `max_tokens_per_run = 16384` ensures no single agent occupies more than 16K tokens of KV cache, preventing runaway context growth.

---

## ⚖️ WORKLOAD TYPES — MIXED BATCH

### The UI can dynamically manage multiple batch classes
Multiple concurrent job categories can be defined with different prompt/output lengths. The Monte Carlo simulation treats all classes as part of the combined workload.

> **Example:** Define class A (`prompt = 500, output = 200, jobs = 30`) for short classification tasks and class B (`prompt = 4000, output = 1000, jobs = 5`) for document summarization — the planner simulates both simultaneously.

---

## 🔀 WORKLOAD TYPES — HYBRID

### Batch section — Prompt tokens
The average prompt length for the batch portion (e.g. 1500). This is the initial context for the batch workload.

> **Example:** `1500` tokens representing a product description being processed in a batch pipeline.

### Batch section — Max output tokens
The maximum response length for the batch portion (e.g. 500).

> **Example:** A batch that generates short summaries may cap at `500` tokens per response.

### Batch section — Concurrent batch jobs
The number of simultaneously running batch jobs in the hybrid workload (e.g. 5).

> **Example:** 5 background batch summarization jobs running alongside 5 interactive agents.

### Agentic section — Initial prompt tokens
The agent's initial context in tokens (e.g. 1000), independent of the batch context.

> **Example:** An agent handling customer support tickets starts with a `1000`-token system prompt describing escalation rules.

### Agentic section — Concurrent agents
The number of simultaneously running agents in the hybrid workload (e.g. 5).

> **Example:** 5 live customer-facing agents running concurrently while batch jobs process in the background.

### Agentic section — Max turn
The maximum number of turns for the agentic portion (e.g. 8).

> **Example:** Support agents are capped at 8 turns before escalating to a human operator.

### Agentic section — Tool call probability (%)
The probability of a tool call in agentic steps (e.g. 70%).

> **Example:** Support agents call CRM lookup tools ~70% of the time per turn.

### Agentic section — Tool result tokens
The average tool response length in tokens (e.g. 500).

> **Example:** A CRM lookup may return a customer history record of ~500 tokens.

### Agentic section — Assistant response tokens
The average agent response length (e.g. 250).

> **Example:** Short acknowledgment messages or clarifying questions averaging 250 tokens.

### Agentic section — Stop probability (%)
The agent's stop probability per turn (e.g. 20%).

> **Example:** A 20% stop probability means the average session lasts ~5 turns before resolution.

### Agentic section — Max tokens / run
The maximum tokens for an agent run (e.g. 16384).

> **Example:** `16384` ensures interactive agents don't consume more than half a 32K context window.

---

## 👥 WORKLOAD TYPES — HUMAN IN THE LOOP

### System prompt tokens
The length of the persistent system message throughout the entire session, in tokens (e.g. 2200). This is always present in the context.

> **Example:** A detailed assistant persona with instructions, formatting rules, and tool descriptions may occupy `2200` tokens.

### Concurrent sessions
The number of simultaneously active human-in-the-loop sessions (e.g. 5). More sessions = more KV cache required.

> **Example:** A customer service platform supporting 100 concurrent live chats needs to plan KV cache for all 100 sessions simultaneously.

### Min rounds
The minimum number of interaction rounds in the session (e.g. 5). One round = one user message + one assistant response.

> **Example:** A sales conversation rarely ends in fewer than 5 rounds; setting `min_rounds = 5` ensures the simulation reflects this.

### Max rounds
The maximum number of rounds in the session (e.g. 30). This caps the context length.

> **Example:** A session capped at `max_rounds = 30` prevents the KV cache from growing beyond 30 × (user + assistant) token pairs.

### User message tokens
The average user message length in tokens (e.g. 250). Added to the session context after each round.

> **Example:** A user typing a detailed technical question might produce `250–400` tokens per message.

### Assistant response tokens
The average assistant response length in tokens (e.g. 400). This grows the session's KV cache each round.

> **Example:** Detailed code explanations or multi-step instructions might average `600–800` tokens per response.

### Tool call probability (%)
The probability (in percent) that the assistant calls a tool in a response (e.g. 60%).

> **Example:** A developer assistant that queries documentation or runs code snippets may call tools ~60% of the time.

### Tool call / round
The average number of tool calls per round (e.g. 2).

> **Example:** If the assistant calls a search tool and a code execution tool in the same turn, set `tool_calls_per_round = 2`.

### Tool output tokens
The average tool response length in tokens (e.g. 650). Added to the context.

> **Example:** A code execution result with stdout and stderr might return ~650 tokens.

### Context summarization
Boolean: whether dynamic context summarization is performed during the session to manage long conversations. Yes/No.

> **Example:** Enable for multi-hour support sessions where context would otherwise exceed the model's context window.

### Summarization trigger
The token count above which summarization is triggered (e.g. 16384). When the context exceeds this value, summarization occurs.

> **Example:** With `summarization_trigger = 16384`, a session accumulating more than 16K tokens is automatically compressed before the next turn.

### Compression ratio
The percentage of the original context retained after summarization (e.g. 0.25 = 75% reduction). Lower value = more aggressive compression.

> **Example:** `compression_ratio = 0.25` means a 20K-token context is compressed to ~5K tokens, freeing significant KV cache space.

### Old turn pruning
Boolean: the system removes older turns to handle excessively long sessions. Yes/No.

> **Example:** Enable when users have very long sessions (e.g. multi-day projects) and the context cannot be compressed enough by summarization alone.

### Keep last N turns
The number of most recent turns to retain after pruning (e.g. 10). Older turns beyond this count are dropped.

> **Example:** `keep_last_n_turns = 10` retains the last 10 exchanges; everything older is discarded to stay within the context limit.

### Tool output truncation
Boolean: whether tool outputs are truncated to a maximum length. Helps reduce context bloat caused by very long tool responses. Yes/No.

> **Example:** Enable when database queries or web scraping tools can return arbitrarily long results that would flood the context.

### Tool truncation limit
The maximum length of tool outputs in tokens (e.g. 1200). Longer outputs are truncated.

> **Example:** `tool_truncation_limit = 1200` ensures no single tool call adds more than 1200 tokens to the context, regardless of actual output size.

### Max context window
The maximum context length (in tokens) supported and used by the model (e.g. 32768). This is the absolute upper limit.

> **Example:** If the model supports 128K tokens but you set `max_context_window = 32768`, the planner will not allow any session to grow beyond 32K tokens.

---

## 🎲 MONTE CARLO

### Monte Carlo enabled
Boolean: whether the stochastic simulation run is enabled. Controlled with a Yes/No toggle.

> **Example:** Enable to get P50/P95/P99 memory and latency estimates instead of a single deterministic value.

### Iterations
The number of Monte Carlo simulation iterations (e.g. 10000). More iterations = higher accuracy, but longer runtime.

> **Example:** `iterations = 1000` is fast (~seconds) but has higher variance; `iterations = 50000` gives stable percentile estimates at the cost of minutes of compute.

### Random seed
The seed for the random number generator, ensuring reproducible runs (e.g. 42). The same seed always produces the same output.

> **Example:** Use `seed = 42` during development for reproducibility; remove the seed (or randomize it) for production capacity planning to avoid seed-dependent bias.

---

## 📋 BLUEPRINT WORKLOAD COMPOSER

### Scenario name
The identifier for the blueprint workload simulation run.

> **Example:** `rag-ecommerce-prod-h100-fp16` describes the blueprint type, domain, environment, hardware, and precision at a glance.

### Blueprint
The blueprint selected from the catalogue (e.g. `rag_pipeline`, `multi_agent_orchestration`). This defines the roles of LLM models and how they are connected.

> **Example:** `multi_agent_orchestration` sets up an orchestrator model directing several specialist sub-agents, each with their own context and token budgets.

### Template
The template used for the given blueprint, containing specific parameter sets (e.g. `ecommerce_product_search`).

> **Example:** `ecommerce_product_search` pre-fills typical retrieval chunk sizes, reranker token budgets, and generation lengths for an e-commerce search use case.

### Target language
The linguistic profile of the workload, affecting average token count estimates.

> **Example:** Switching from `English` to `Hungarian` increases estimated token counts for equivalent content, raising projected memory usage.

### Language share
The proportion of traffic that applies to the given language setting (e.g. 0.8 = 80% in the primary language, 20% other).

> **Example:** `language_share = 0.7` means 70% of requests are processed with the Hungarian token overhead; the remaining 30% use English estimates.

### Arrival rate / sec
The arrival rate per second (e.g. 3 users/second). This is the intensity of the workload.

> **Example:** `arrival_rate = 10` represents a peak load of 10 new requests per second; the planner calculates the required throughput and memory to sustain this.

### SLA p95 latency (ms)
The target maximum latency at the 95th percentile, in milliseconds (e.g. 2200ms). The system tries to meet this target.

> **Example:** `sla_p95 = 1500ms` is a tight interactive SLA; `sla_p95 = 10000ms` is acceptable for background batch jobs.

### Monte Carlo iterations
The number of Monte Carlo iterations used for the blueprint simulation (e.g. 40). Lower than full MC because blueprints run faster.

> **Example:** `iterations = 40` is sufficient for quick blueprint comparisons; increase to `200+` for final production planning.

### Time horizon (sec)
The simulated time interval in seconds (e.g. 300 seconds = 5 minutes). Longer = more load is accumulated.

> **Example:** `time_horizon = 3600` simulates a full hour of sustained load, revealing memory pressure from long-lived agent sessions.

### Planning profile
The capacity planning strategy: Baseline, 24/7 safe (high reserve), or High throughput (tight packing). This affects reserved capacity.

> **Example:** Use `24/7 safe` for production systems requiring high availability; use `High throughput` for batch processing where occasional OOM is acceptable.

---

## 📚 ADVANCED TEMPLATES

### Scenario name
The identifier for the template workload simulation run.

> **Example:** `retrieval-rerank-prod-a100-bf16-v3`

### Template
The template selected from the template library (e.g. `retrieval_rerank_generation`, `agentic_workflow`, `batch_document`). This defines roles and static load profiles.

> **Example:** `retrieval_rerank_generation` models a three-stage RAG pipeline: retrieval (embedding model), reranking (cross-encoder), and generation (LLM).

### Target language
The language setting for the workload (Hungarian, English, German).

> **Example:** `German` increases token estimates slightly compared to English due to compound words.

### Language share
The proportion of the workload that applies to the given language setting (between 0 and 1).

> **Example:** `language_share = 0.9` for a primarily German-language deployment, with `0.1` for English edge cases.

### SLA p95 latency (ms)
The target 95th percentile latency (e.g. 2200ms).

> **Example:** A document processing pipeline might tolerate `sla_p95 = 5000ms`; a real-time chat interface needs `< 1000ms`.

### Monte Carlo iterations
The number of Monte Carlo runs for the template simulation (e.g. 40).

> **Example:** Increase to `500` when using template simulations for final hardware procurement decisions.

### Time horizon (sec)
The simulated time frame in seconds (e.g. 300).

> **Example:** `time_horizon = 600` (10 minutes) is useful for spotting memory leaks or context growth that only manifest after sustained load.

### Planning profile
The capacity planning profile: Baseline, 24/7 safe, or High throughput.

> **Example:** `High throughput` minimizes reserved headroom to maximize GPU utilization in cost-sensitive batch pipelines.

### Hardware, Software stack, Deployment profile
The specific infrastructure and software layer selection.

> **Example:** `nvidia_dgx_h100_8gpu` + `nvidia_vllm_cuda` + `bare_metal` is a typical high-performance production stack for large model inference.

---

## 📊 RESULTS AND ANALYSIS

### JSON import
Upload result files for analysis. The UI automatically processes the `hardware_catalog_id`, `model_id`, `software_stack_id`, and `deployment_profile_id` fields.

> **Example:** Import `run_2024-11-01_llama3-70b.json` to compare it side-by-side with a newer quantized run in the Results tab.

### Filter options
- **Result type**: Filter by Workload simulation or Capacity/memory analysis
- **Hardware**: Filter by a specific hardware device
- **Model**: Filter by a specific model
- **Risk level**: Filter by LOW, MODERATE, HIGH, or CRITICAL risk

> **Example:** Filter by `Risk level = HIGH` to immediately surface all configurations where the planner estimated less than 10% memory headroom under peak load.

---

**Note:** All parameters in the Capacity Planner UI expect inputs that conform to international standards. The simulation engine processes configurations in real time and provides immediate feedback on compatibility and resource requirements.
