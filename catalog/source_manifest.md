# Source manifest for seed catalogs

This bundle prepares curated seed data for integrating dynamic Hugging Face model loading with static infrastructure catalogs.

## Hardware and GPU sources
- NVIDIA H200 official product page — H200 141 GB HBM3e, 4.8 TB/s: https://www.nvidia.com/en-us/data-center/h200/
- NVIDIA RTX PRO 6000 Blackwell Server Edition — 96 GB GDDR7, 1597 GB/s: https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/
- NVIDIA DGX B200 official page — 8x B200, 1,440 GB total GPU memory, 14.4 TB/s aggregate NVLink: https://www.nvidia.com/en-us/data-center/dgx-b200/
- NVIDIA DGX B300 official page — 8x Blackwell Ultra SXM, 14.4 TB/s aggregate NVLink: https://www.nvidia.com/en-us/data-center/dgx-b300/
- NVIDIA HGX platform page — HGX B200 / HGX B300 platform properties: https://www.nvidia.com/en-us/data-center/hgx/
- NVIDIA MIG User Guide — supported GPUs and profiles for H200 / B200 / RTX PRO 6000 BE: https://docs.nvidia.com/datacenter/tesla/mig-user-guide/
- Dell PowerEdge GPU matrix: https://www.delltechnologies.com/asset/nl-nl/products/servers/briefs-summaries/poweredge-server-gpu-matrix.pdf
- Dell PowerEdge XE9680 spec sheet: https://www.delltechnologies.com/asset/en-in/products/servers/technical-support/poweredge-xe9680-spec-sheet.pdf
- Dell PowerEdge XE7740 technical guide: https://www.delltechnologies.com/asset/en-nz/products/servers/technical-support/poweredge-xe7740-technical-guide.pdf
- Dell PowerEdge acceleration brief (R770 / XE8640 / XE9680 context): https://www.delltechnologies.com/asset/es-es/products/servers/briefs-summaries/poweredge-acceleration-innovate-faster-for-ai.pdf
- HPE DL380a Gen12 page / QuickSpecs: https://www.hpe.com/emea_europe/en/compute/hpe-proliant-compute/dl380a-gen12.html and https://www.hpe.com/psnow/doc/a00047453enw
- HPE DL385 Gen11 page / QuickSpecs: https://www.hpe.com/emea_europe/en/compute/hpe-proliant-compute/dl385-gen11.html and https://www.hpe.com/psnow/doc/a50004300enw
- HPE NVIDIA accelerators QuickSpecs: https://www.hpe.com/psnow/doc/c04123180
- Supermicro GPU qualified platform list: https://www.supermicro.com/en/support/resources/gpu
- Supermicro NVIDIA PCIe GPU systems page: https://www.supermicro.com/en/accelerators/nvidia/pcie-gpu
- AMD MI300X official product page and datasheet: https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html and https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf

## Runtime / software sources
- vLLM quickstart / install docs: https://docs.vllm.ai/en/v0.12.0/getting_started/quickstart/
- vLLM optimization docs: https://docs.vllm.ai/en/stable/configuration/optimization/
- NVIDIA TensorRT-LLM docs: https://docs.nvidia.com/tensorrt-llm/index.html
- NVIDIA TensorRT support matrix: https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html
- NVIDIA NIM LLM support matrix: https://docs.nvidia.com/nim/large-language-models/latest/support-matrix.html
- NVIDIA AI Enterprise support matrix: https://docs.nvidia.com/ai-enterprise/release-8/latest/support/support-matrix-8/8.0.html
- AMD ROCm vLLM benchmark docker docs: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/vllm.html
- AMD ROCm MI300X vLLM benchmark docs: https://rocm.docs.amd.com/en/docs-6.3.1/how-to/rocm-for-ai/inference/vllm-benchmark.html
- VMware/NVIDIA vGPU vs MIG performance paper: https://www.vmware.com/techpapers/2022/vgpu-vs-mig-perf.html

## Model source / ingest docs
- Hugging Face HfApi client docs: https://huggingface.co/docs/huggingface_hub/package_reference/hf_api
- Hugging Face model cards docs: https://huggingface.co/docs/hub/model-cards

## Notes
- This is a curated starter set, not an exhaustive catalog.
- Hardware entries are intentionally normalized around 2 / 4 / 8 GPU serving nodes and reference platforms.
- B200 / B300 are included primarily via NVIDIA DGX/HGX reference systems to cover the newest Blackwell classes cleanly.
