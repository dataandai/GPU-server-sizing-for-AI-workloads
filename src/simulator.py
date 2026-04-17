"""
Main simulator orchestrator.
Loads scenarios, runs simulations, collects results.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

from .config import SimulationConfig, SimulationResult, HardwareProfileName
from .workload_simulation import (
    WorkloadSimulationEngine,
    WorkloadSimulationResult,
    is_workload_simulation_file,
    load_workload_simulation,
)
from .workloads import load_scenario, load_scenarios_multi_hardware
from .monte_carlo import MonteCarloEngine


def run_single(config: SimulationConfig) -> SimulationResult:
    """Run a single simulation with the given configuration."""
    engine = MonteCarloEngine(config)
    return engine.run()


def run_scenario_file(path: str) -> SimulationResult | WorkloadSimulationResult:
    """Load and run a single scenario from a YAML file."""
    if is_workload_simulation_file(path):
        spec = load_workload_simulation(path)
        return WorkloadSimulationEngine(spec).run()
    config = load_scenario(path)
    return run_single(config)


def run_scenario_all_hardware(path: str) -> list[SimulationResult | WorkloadSimulationResult]:
    """Load a scenario and run it across all 4 hardware profiles.

    Workload-simulation YAML files are already deployment-bound, so they are run once.
    """
    if is_workload_simulation_file(path):
        return [run_scenario_file(path)]
    configs = load_scenarios_multi_hardware(path)
    results = []
    for cfg in configs:
        engine = MonteCarloEngine(cfg)
        results.append(engine.run())
    return results


def run_scenario_directory(
    directory: str,
    hardware_filter: Optional[HardwareProfileName] = None,
) -> list[SimulationResult | WorkloadSimulationResult]:
    """
    Run all YAML scenarios in a directory.
    If hardware_filter is None, runs each scenario on all 4 hardware profiles.
    """
    scenario_dir = Path(directory)
    results = []

    for yaml_path in sorted(scenario_dir.glob("*.yaml")):
        if hardware_filter:
            config = load_scenario(str(yaml_path))
            config.hardware_profile = hardware_filter
            if hardware_filter in (
                HardwareProfileName.H200_8GPU,
                HardwareProfileName.RTX6000_8GPU,
            ):
                config.tensor_parallel_degree = 8
            else:
                config.tensor_parallel_degree = 4
            config.scenario_name = f"{config.scenario_name}__{hardware_filter.value}"
            results.append(run_single(config))
        else:
            results.extend(run_scenario_all_hardware(str(yaml_path)))

    return results
