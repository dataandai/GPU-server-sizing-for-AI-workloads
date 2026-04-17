# Consistency Improvements — Audit Phase 3

This phase extends the audit-driven hardening work with four focused improvements:

## 1. Procurement and risk decision-table coverage

Added direct boundary tests for the workload risk classifier and procurement thresholds:
- critical / high / medium risk boundaries in `WorkloadSimulationEngine._assess_risk()`
- exact procurement margin behavior around the `UPSIZE_REQUIRED` threshold
- explicit non-risk behavior at exactly 90% utilization
- target-less workload fallback behavior

## 2. More family-specific reporting coverage

Added domain-specific detailed-metric tests for:
- route optimization
- predictive forecasting / maintenance analytics
- recommendation / ranking service

This strengthens the user-facing reporting surface so that family-specific cards are not only indirectly covered.

## 3. Legacy memory engine smoke coverage

Added smoke and deterministic tests for the legacy VRAM / KV-cache path:
- fixed batch legacy scenario
- HITL legacy scenario
- deterministic `KVCacheModel` behavior for fixed, mixed, hybrid, agentic, and HITL workloads

Important note: the agentic and HITL peak token tests intentionally lock in the current **pre-governance peak** semantics. The peak is recorded before context cap / hard cap governance is applied, which is conservative and should remain explicit.

## 4. Optional CLI catalog audit

Added `run_all.py --catalog-audit` and `run_all.py --strict-catalog-audit`.

Purpose:
- run semantic catalog cross-reference validation without starting simulations
- optionally fail fast when integrity errors are present

This supports release-time and CI-time consistency checks for catalog-driven behavior.
