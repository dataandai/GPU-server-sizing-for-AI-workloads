"""
KV Cache growth model for different workload types.

Models how KV cache grows during inference:
- Fixed batch: linear growth (prompt + output tokens)
- Agentic: stochastic growth with tool calls, multi-turn, summarization
- Human-in-the-loop: continuous growth from long-lived interactive sessions

Simplifying Assumptions:
1. KV cache size = sum of all sequence lengths × per-token KV cost
2. "Sequence length" = total number of tokens in the context window
3. For batch inference, we model the PEAK state (all sequences at max length)
4. For agentic, we simulate turn-by-turn growth with probabilistic events
5. For HITL, we simulate round-by-round growth with user/assistant/tool tokens
6. No prefix caching / shared prompt deduplication modeled (conservative)
7. PagedAttention block alignment overhead ignored (adds ~1-3% in practice)
"""

from __future__ import annotations
import numpy as np
from .config import (
    DistributionSpec, FixedBatchWorkload, MixedBatchWorkload,
    AgenticWorkload, HybridWorkload, HumanInTheLoopWorkload,
)
from .distributions import DistributionSampler


class KVCacheModel:
    """Models KV cache token accumulation for different workload types."""

    def __init__(self, sampler: DistributionSampler):
        self.sampler = sampler

    def simulate_fixed_batch(self, workload: FixedBatchWorkload) -> np.ndarray:
        """
        Simulate peak KV cache token count for each sequence in a fixed batch.

        Each sequence's peak cache size = prompt_tokens + output_tokens.
        For deterministic mode, uses fixed values.
        For MC mode, samples from distributions.

        Returns: array of per-sequence token counts, shape (num_parallel_jobs,)
        """
        n = workload.num_parallel_jobs
        prompts = self.sampler.sample_int(workload.prompt_tokens, n)
        outputs = self.sampler.sample_int(workload.max_output_tokens, n)
        return prompts + outputs

    def simulate_mixed_batch(self, workload: MixedBatchWorkload) -> np.ndarray:
        """
        Simulate peak KV cache for a mixed batch with multiple job classes.

        Returns: array of per-sequence token counts
        """
        all_tokens = []
        for cls in workload.classes:
            prompts = self.sampler.sample_int(cls.prompt_tokens, cls.count)
            outputs = self.sampler.sample_int(cls.max_output_tokens, cls.count)
            all_tokens.append(prompts + outputs)
        return np.concatenate(all_tokens) if all_tokens else np.array([], dtype=int)

    def simulate_agentic_single(self, workload: AgenticWorkload) -> int:
        """
        Simulate KV cache growth for a SINGLE agentic run.

        Turn-by-turn simulation:
        1. Start with initial prompt tokens
        2. Each turn:
           a. Model generates assistant response tokens
           b. With probability tool_call_probability, a tool call is made
           c. Tool call adds tool_result_tokens to the context
           d. With probability stop_probability_per_turn, the agent stops
           e. With probability summarization_probability, context is compressed

        Returns: peak total token count for this agent run
        """
        # Initial prompt
        total_tokens = self.sampler.sample_single_int(workload.initial_prompt_tokens)
        peak_tokens = total_tokens

        for turn in range(workload.max_turns_per_run):
            # Assistant response
            response_tokens = self.sampler.sample_single_int(workload.assistant_response_tokens)
            total_tokens += response_tokens

            # Tool call?
            if self.sampler.rng.random() < workload.tool_call_probability:
                tool_tokens = self.sampler.sample_single_int(workload.tool_result_tokens)
                total_tokens += tool_tokens

            # Update peak
            peak_tokens = max(peak_tokens, total_tokens)

            # Context cap
            if workload.enable_max_context_cap and total_tokens > workload.max_context_cap:
                total_tokens = workload.max_context_cap
                peak_tokens = max(peak_tokens, total_tokens)

            # Context reset (e.g., sliding window or hard reset)
            if workload.enable_context_reset and total_tokens > workload.context_reset_threshold:
                # [ASSUMPTION] Reset keeps last 50% of context
                total_tokens = workload.context_reset_threshold // 2

            # Summarization
            if workload.summarization_probability > 0:
                if self.sampler.rng.random() < workload.summarization_probability:
                    total_tokens = int(total_tokens * workload.summarization_compression_ratio)

            # Stop check
            if self.sampler.rng.random() < workload.stop_probability_per_turn:
                break

            # Hard cap on total tokens per run
            if total_tokens >= workload.max_total_tokens_per_run:
                total_tokens = workload.max_total_tokens_per_run
                peak_tokens = max(peak_tokens, total_tokens)
                break

        return peak_tokens

    def simulate_agentic(self, workload: AgenticWorkload) -> np.ndarray:
        """
        Simulate peak KV cache for all parallel agentic runs.

        Returns: array of per-agent peak token counts, shape (num_parallel_agents,)
        """
        peaks = np.array([
            self.simulate_agentic_single(workload)
            for _ in range(workload.num_parallel_agents)
        ])
        return peaks

    def simulate_human_in_the_loop_single(self, workload: HumanInTheLoopWorkload) -> int:
        """
        Simulate KV cache growth for a SINGLE human-in-the-loop coding session.

        Models the real-world pattern of an interactive development session:
        1. Large system prompt is always present (rules, repo context, files)
        2. Each round adds:
           a. User message (code, question, diff, clarification)
           b. Assistant response (explanation, generated code)
           c. With probability, N tool calls are made (build, test, file read)
           d. Each tool output adds to the context
        3. Context governance strategies can reduce growth:
           - Summarization: compress old context when threshold exceeded
           - Old turn pruning: drop turns beyond last N
           - Tool output truncation: cap individual tool outputs
           - Hard context cap: vLLM max_model_len equivalent

        Returns: peak total token count for this session
        """
        # System prompt — always present, never shrinks
        system_tokens = self.sampler.sample_single_int(workload.system_prompt_tokens)
        total_tokens = system_tokens
        peak_tokens = total_tokens

        # Determine session length (clamped between min and max rounds)
        raw_rounds = self.sampler.sample_single_int(workload.rounds_distribution)
        num_rounds = max(workload.min_rounds, min(workload.max_rounds, raw_rounds))

        # Track turn-level tokens for pruning simulation
        turn_token_history = []

        for round_idx in range(num_rounds):
            round_tokens = 0

            # User message
            user_tokens = self.sampler.sample_single_int(workload.user_message_tokens)
            round_tokens += user_tokens

            # Assistant response
            response_tokens = self.sampler.sample_single_int(workload.assistant_response_tokens)
            round_tokens += response_tokens

            # Tool calls (multiple possible per round)
            if self.sampler.rng.random() < workload.tool_call_probability:
                num_tools = self.sampler.sample_single_int(workload.num_tool_calls_per_round)
                num_tools = max(1, num_tools)
                for _ in range(num_tools):
                    tool_tokens = self.sampler.sample_single_int(workload.tool_output_tokens)
                    # Tool output truncation governance
                    if workload.enable_tool_output_truncation:
                        tool_tokens = min(tool_tokens, workload.tool_output_max_tokens)
                    round_tokens += tool_tokens

            total_tokens += round_tokens
            turn_token_history.append(round_tokens)

            # Update peak BEFORE governance (peak represents worst-case moment)
            peak_tokens = max(peak_tokens, total_tokens)

            # --- Context Governance ---

            # Old turn pruning: keep only last N turns
            if workload.enable_old_turn_pruning and len(turn_token_history) > workload.pruning_keep_last_n_turns:
                excess_turns = len(turn_token_history) - workload.pruning_keep_last_n_turns
                pruned_tokens = sum(turn_token_history[:excess_turns])
                total_tokens -= pruned_tokens
                turn_token_history = turn_token_history[excess_turns:]

            # Context summarization: compress when context exceeds threshold
            if workload.enable_context_summarization:
                if total_tokens > workload.summarization_trigger_tokens:
                    # Compress the non-system-prompt portion
                    compressible = total_tokens - system_tokens
                    compressed = int(compressible * workload.summarization_compression_ratio)
                    total_tokens = system_tokens + compressed
                    # Reset turn history after summarization
                    turn_token_history = [compressed]

            # Hard context cap (vLLM max_model_len)
            if workload.enable_hard_context_cap and total_tokens > workload.hard_context_cap:
                total_tokens = workload.hard_context_cap
                peak_tokens = max(peak_tokens, total_tokens)

            # Human stops the session
            if self.sampler.rng.random() < workload.session_stop_probability_per_round:
                break

        return peak_tokens

    def simulate_human_in_the_loop(self, workload: HumanInTheLoopWorkload) -> np.ndarray:
        """
        Simulate peak KV cache for all concurrent HITL sessions.

        Returns: array of per-session peak token counts, shape (num_concurrent_sessions,)
        """
        peaks = np.array([
            self.simulate_human_in_the_loop_single(workload)
            for _ in range(workload.num_concurrent_sessions)
        ])
        return peaks

    def simulate_hybrid(self, workload: HybridWorkload) -> np.ndarray:
        """
        Simulate hybrid workload: fixed batch + agentic running concurrently.

        Returns: concatenated array of all per-sequence peak token counts
        """
        parts = []
        if workload.fixed_batch:
            parts.append(self.simulate_fixed_batch(workload.fixed_batch))
        if workload.agentic:
            parts.append(self.simulate_agentic(workload.agentic))
        return np.concatenate(parts) if parts else np.array([], dtype=int)

    def get_total_active_tokens(self, token_counts: np.ndarray) -> int:
        """
        Get total active tokens across all sequences.
        This represents the PEAK concurrent KV cache state.
        """
        return int(np.sum(token_counts))

    def get_max_sequence_length(self, token_counts: np.ndarray) -> int:
        """Get the longest individual sequence."""
        return int(np.max(token_counts)) if len(token_counts) > 0 else 0
