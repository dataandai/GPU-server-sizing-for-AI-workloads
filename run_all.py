#!/usr/bin/env python3
"""Main CLI entry point for the LLM VRAM / KV-cache simulator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.catalog_loader import (
    list_blueprint_records,
    list_deployment_profile_records,
    list_hardware_records,
    list_software_stack_records,
    validate_catalog_integrity,
)
from src.reporting import (
    format_detailed_report,
    format_hardware_comparison,
    format_summary_table,
    results_to_json,
)
from src.report_export import export_customer_report_html
from src.pdf_export import export_customer_report_pdf
from src.file_naming import default_json_output_path
from src.simulator import run_scenario_all_hardware, run_scenario_file
from src.workload_simulation import is_workload_simulation_file


def _print_catalog(records: list[dict], title: str, fields: list[str]) -> None:
    print(title)
    print("=" * len(title))
    for record in records:
        pieces = [str(record.get(field, "-")) for field in fields]
        print(" • " + " | ".join(pieces))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM VRAM & KV Cache Monte Carlo Simulator"
    )
    parser.add_argument(
        "scenarios",
        nargs="*",
        default=None,
        help="Specific scenario YAML files to run (default: all in scenarios/)",
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON to stdout")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Write machine-readable JSON results to this file. If omitted, a readable model_hardware_timestamp name is used under output/results/.",
    )
    parser.add_argument(
        "--no-json-write",
        action="store_true",
        help="Do not write results.json to disk",
    )
    parser.add_argument("--detailed", action="store_true", help="Print detailed reports")
    parser.add_argument("--no-comparison", action="store_true", help="Skip hardware comparison analysis")
    parser.add_argument("--list-hardware", action="store_true", help="List curated hardware catalog entries")
    parser.add_argument("--list-stacks", action="store_true", help="List curated software stack profiles")
    parser.add_argument("--list-deployments", action="store_true", help="List deployment / virtualization profiles")
    parser.add_argument("--list-blueprints", action="store_true", help="List curated NVIDIA blueprint records")
    parser.add_argument("--catalog-audit", action="store_true", help="Run semantic catalog integrity checks before any simulation work.")
    parser.add_argument("--strict-catalog-audit", action="store_true", help="Fail with exit code 2 if semantic catalog integrity errors are found.")
    parser.add_argument("--report-html", nargs="?", const="auto", default=None, help="Export a customer-facing HTML report. Optionally pass an output path.")
    parser.add_argument("--report-pdf", nargs="?", const="auto", default=None, help="Export a customer-facing PDF report. Optionally pass an output path.")
    parser.add_argument("--report-from-json", default=None, help="Build report directly from an existing workload result JSON file.")
    args = parser.parse_args()

    if args.list_hardware:
        _print_catalog(list_hardware_records(), "Curated hardware catalog", ["id", "display_name", "gpu_count", "gpu_id"])
        return
    if args.list_stacks:
        _print_catalog(list_software_stack_records(), "Curated software stacks", ["id", "framework", "vendor_focus"])
        return
    if args.list_deployments:
        _print_catalog(list_deployment_profile_records(), "Deployment profiles", ["id", "mode", "gpu_partitioning"])
        return
    if args.list_blueprints:
        _print_catalog(list_blueprint_records(), "NVIDIA blueprint catalog", ["blueprint_id", "name", "category"])
        return

    if args.catalog_audit or args.strict_catalog_audit:
        errors, warnings = validate_catalog_integrity()
        print("Catalog integrity audit")
        print("=======================")
        print(f"Errors:   {len(errors)}")
        print(f"Warnings: {len(warnings)}")
        for entry in errors[:25]:
            print(f"  ERROR: {entry}")
        if len(errors) > 25:
            print(f"  ... {len(errors) - 25} additional error(s) omitted")
        for entry in warnings[:25]:
            print(f"  WARN:  {entry}")
        if len(warnings) > 25:
            print(f"  ... {len(warnings) - 25} additional warning(s) omitted")
        print()
        if args.strict_catalog_audit and errors:
            print("Catalog audit failed in strict mode.")
            sys.exit(2)
        if not args.scenarios and not args.report_from_json and not any((args.json, args.detailed, args.report_html, args.report_pdf)):
            return

    if args.report_from_json:
        result_path = Path(args.report_from_json)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            payload = [payload]
        if args.report_html:
            html_path = Path(f"output/results/{result_path.stem}_report.html") if args.report_html == "auto" else Path(args.report_html)
            export_customer_report_html(payload, html_path)
            print(f"HTML report written to: {html_path}")
        if args.report_pdf:
            pdf_path = Path(f"output/results/{result_path.stem}_report.pdf") if args.report_pdf == "auto" else Path(args.report_pdf)
            export_customer_report_pdf(payload, pdf_path)
            print(f"PDF report written to: {pdf_path}")
        if args.report_html or args.report_pdf:
            return

    if args.scenarios:
        scenario_files = []
        for s in args.scenarios:
            p = Path(s)
            if p.is_file():
                scenario_files.append(p)
            else:
                scenario_files.extend(Path(".").glob(s))
    else:
        scenario_dir = Path(__file__).parent / "scenarios"
        scenario_files = sorted(scenario_dir.glob("*.yaml"))

    if not scenario_files:
        print("No scenario files found!")
        sys.exit(1)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  LLM VRAM & KV Cache Monte Carlo Simulator                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\nFound {len(scenario_files)} scenario(s) to run...")
    if any(not is_workload_simulation_file(str(sf)) for sf in scenario_files):
        print("⚠ Legacy memory-simulation scenarios detected. These use the classic VRAM/KV-cache engine, not the benchmark-calibrated workload simulator.")
    print()

    all_results = []
    for sf in scenario_files:
        print(f"  ▸ Running: {sf.name} ...", end=" ", flush=True)
        try:
            results = run_scenario_all_hardware(str(sf))
            all_results.extend(results)
            critical = [r for r in results if getattr(r, "qualitative_risk_level", "") == "CRITICAL"]
            if critical:
                print(f"done ({len(results)} configs, ⚠ {len(critical)} CRITICAL)")
            else:
                print(f"done ({len(results)} configs)")
        except Exception as exc:
            print(f"FAILED: {exc}")
            continue

    print(f"\nTotal simulation runs: {len(all_results)}\n")

    if not all_results:
        print("No successful simulation results were produced.")
        sys.exit(1)

    json_payload = results_to_json(all_results)
    if not args.no_json_write:
        output_path = Path(args.json_out) if args.json_out else default_json_output_path(all_results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_payload + "\n", encoding="utf-8")
        print(f"JSON results written to: {output_path}")

    if args.report_html:
        html_path = Path(f"output/results/{Path(args.json_out).stem if args.json_out else 'report'}_report.html") if args.report_html == "auto" else Path(args.report_html)
        export_customer_report_html(all_results, html_path)
        print(f"HTML report written to: {html_path}")
    if args.report_pdf:
        pdf_path = Path(f"output/results/{Path(args.json_out).stem if args.json_out else 'report'}_report.pdf") if args.report_pdf == "auto" else Path(args.report_pdf)
        export_customer_report_pdf(all_results, pdf_path)
        print(f"PDF report written to: {pdf_path}")

    if args.json:
        print(json_payload)
        return

    print("=" * 80)
    print("  SUMMARY TABLE")
    print("=" * 80)
    print()
    print(format_summary_table(all_results))
    print()

    if args.detailed:
        for result in all_results:
            print(format_detailed_report(result))

    if not args.no_comparison:
        print(format_hardware_comparison(all_results))


if __name__ == "__main__":
    main()
