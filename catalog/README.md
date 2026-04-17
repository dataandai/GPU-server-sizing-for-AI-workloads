# Catalog bundle

This folder contains a static seed catalog for infrastructure plus a dynamic ingest contract for Hugging Face models.

## Files
- `gpu_catalog.json` — normalized GPU-level records
- `hardware_catalog.json` — 2/4/8 GPU vendor node records and DGX references
- `software_stacks.json` — curated serving/runtime profiles
- `deployment_profiles.json` — execution environment and virtualization / MIG modes
- `hf_model_ingest_contract.json` — field mapping for dynamic Hugging Face imports
- `source_manifest.md` — primary public sources used for seed preparation

## Intended integration pattern
1. User selects or enters a Hugging Face model ID.
2. Program loads config / card data dynamically and normalizes into current `ModelConfig`.
3. User selects a static `hardware_id`, `gpu_id`, `software_stack_id`, and optionally a `deployment_profile_id`.
4. Simulation converts these normalized records into the existing simulation config.

## Design choices
- Hardware/software remain static and curated for stability.
- Hugging Face models remain dynamic.
- Deployment mode is kept separate from hardware because bare metal vs passthrough vs time-sliced vGPU vs MIG materially changes usable capacity and throughput.
