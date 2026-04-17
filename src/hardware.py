"""
Pre-configured hardware profiles for supported GPU configurations.
"""

from .config import HardwareProfile, HardwareProfileName


HARDWARE_PROFILES: dict[HardwareProfileName, HardwareProfile] = {
    HardwareProfileName.H200_8GPU: HardwareProfile(
        name="8× NVIDIA H200 (HGX/NVLink)",
        num_gpus=8,
        vram_per_gpu_gb=141.0,
        memory_bandwidth_gbps=4800.0,
        interconnect="nvlink",
        notes="HGX platform with NVLink/NVSwitch scale-up fabric. "
              "900 GB/s bidirectional NVLink per GPU. Optimal for TP=8.",
    ),
    HardwareProfileName.H200_4GPU: HardwareProfile(
        name="4× NVIDIA H200 (HGX/NVLink)",
        num_gpus=4,
        vram_per_gpu_gb=141.0,
        memory_bandwidth_gbps=4800.0,
        interconnect="nvlink",
        notes="4-GPU subset of HGX node. NVLink interconnect. "
              "Suitable for TP=4. Tighter VRAM budget.",
    ),
    HardwareProfileName.RTX6000_8GPU: HardwareProfile(
        name="8× NVIDIA RTX PRO 6000 Blackwell Server Edition",
        num_gpus=8,
        vram_per_gpu_gb=96.0,
        memory_bandwidth_gbps=1597.0,
        interconnect="pcie5",
        notes="PCIe 5.0 x16 interconnect (~64 GB/s per direction). "
              "GDDR7 ECC memory. MIG support up to 4×24GB (not used for full model). "
              "Blackwell architecture with native FP4 Tensor Core support.",
    ),
    HardwareProfileName.RTX6000_4GPU: HardwareProfile(
        name="4× NVIDIA RTX PRO 6000 Blackwell Server Edition",
        num_gpus=4,
        vram_per_gpu_gb=96.0,
        memory_bandwidth_gbps=1597.0,
        interconnect="pcie5",
        notes="PCIe 5.0 x16. 4-GPU config with TP=4. "
              "384 GB total — very tight for 235B model. "
              "INT4/NVFP4 may be required for this config.",
    ),
}


def get_hardware_profile(name: HardwareProfileName) -> HardwareProfile:
    """Get a hardware profile by name."""
    return HARDWARE_PROFILES[name]
