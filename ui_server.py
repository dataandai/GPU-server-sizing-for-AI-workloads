#!/usr/bin/env python3
"""Small local UI server for running Monte Carlo jobs from ui.html."""

from __future__ import annotations

import json
import os
import traceback
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from src.catalog_loader import (
    get_advanced_workload_template_record,
    get_blueprint_bundle,
    get_workload_simulation_schema,
    list_advanced_workload_template_records,
    list_blueprint_records,
    list_blueprint_template_records,
    list_deployment_profile_records,
    list_hardware_records,
    list_software_stack_records,
    list_ui_help_records,
)
from src.file_naming import build_base_name_from_config, slugify
from src.model_loader import load_model_config_from_huggingface
from src.blueprint_adapter import build_blueprint_workload_yaml, build_workload_from_blueprint, list_bindable_roles
from src.template_adapter import build_template_workload_yaml, build_workload_from_template
from src.report_export import export_customer_report_html
from src.pdf_export import export_customer_report_pdf
from src.ui_relevance import build_blueprint_ui_context, build_template_ui_context
from src.model_compatibility import assess_template_model_compatibility, assess_blueprint_model_compatibility, build_role_model_map
from src.reporting import result_to_dict, results_to_json
from src.simulator import run_single
from src.workload_simulation import WorkloadSimulationEngine, is_workload_simulation_payload
from src.workloads import load_scenario
import yaml

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_SCENARIOS = REPO_ROOT / "output" / "generated_scenarios"
OUTPUT_RESULTS = REPO_ROOT / "output" / "results"


class UIServerHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        clean = parsed.path or "/"
        if clean == "/":
            clean = "/ui.html"
        target = (REPO_ROOT / clean.lstrip("/")).resolve()
        if REPO_ROOT not in target.parents and target != REPO_ROOT:
            return str(REPO_ROOT / "ui.html")
        return str(target)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json({"ok": True})
            return
        if parsed.path == "/api/catalog/summary":
            self._send_json(
                {
                    "ok": True,
                    "counts": {
                        "hardware": len(list_hardware_records()),
                        "software_stacks": len(list_software_stack_records()),
                        "deployment_profiles": len(list_deployment_profile_records()),
                        "blueprints": len(list_blueprint_records()),
                        "blueprint_templates": len(list_blueprint_template_records()),
                        "workload_templates": len(list_advanced_workload_template_records()),
                    }
                }
            )
            return
        if parsed.path == "/api/catalog/hardware":
            self._send_json({"ok": True, "records": list_hardware_records()})
            return
        if parsed.path == "/api/catalog/software-stacks":
            self._send_json({"ok": True, "records": list_software_stack_records()})
            return
        if parsed.path == "/api/catalog/deployments":
            self._send_json({"ok": True, "records": list_deployment_profile_records()})
            return
        if parsed.path == "/api/catalog/blueprints":
            self._send_json({"ok": True, "records": list_blueprint_records()})
            return
        if parsed.path == "/api/catalog/blueprint-templates":
            self._send_json({"ok": True, "records": list_blueprint_template_records()})
            return
        if parsed.path == "/api/catalog/workload-templates":
            self._send_json({"ok": True, "records": list_advanced_workload_template_records()})
            return
        if parsed.path == "/api/catalog/workload-schema":
            self._send_json({"ok": True, "schema": get_workload_simulation_schema()})
            return
        if parsed.path == "/api/ui-help":
            self._send_json({"ok": True, "records": list_ui_help_records()})
            return
        if parsed.path == "/api/ui/template-form":
            query = parse_qs(parsed.query or "")
            template_id = str((query.get("template_id") or [""])[0]).strip()
            if not template_id:
                self._send_json({"ok": False, "error": "Missing template id"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                context = build_template_ui_context(
                    template_id,
                    hardware_catalog_id=str((query.get("hardware_catalog_id") or [""])[0]).strip() or None,
                    software_stack_id=str((query.get("software_stack_id") or [""])[0]).strip() or None,
                    deployment_profile_id=str((query.get("deployment_profile_id") or [""])[0]).strip() or None,
                )
                self._send_json({"ok": True, "context": context})
            except KeyError:
                self.send_error(HTTPStatus.NOT_FOUND, "Template not found")
            return
        if parsed.path == "/api/ui/blueprint-form":
            query = parse_qs(parsed.query or "")
            blueprint_id = str((query.get("blueprint_id") or [""])[0]).strip()
            if not blueprint_id:
                self._send_json({"ok": False, "error": "Missing blueprint id"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                context = build_blueprint_ui_context(
                    blueprint_id,
                    template_id=str((query.get("template_id") or [""])[0]).strip() or None,
                    hardware_catalog_id=str((query.get("hardware_catalog_id") or [""])[0]).strip() or None,
                    software_stack_id=str((query.get("software_stack_id") or [""])[0]).strip() or None,
                    deployment_profile_id=str((query.get("deployment_profile_id") or [""])[0]).strip() or None,
                )
                self._send_json({"ok": True, "context": context})
            except KeyError:
                self.send_error(HTTPStatus.NOT_FOUND, "Blueprint not found")
            return
        if parsed.path.startswith("/api/catalog/blueprint-roles/"):
            blueprint_id = unquote(parsed.path.split("/api/catalog/blueprint-roles/", 1)[1]).strip("/")
            if not blueprint_id:
                self.send_error(HTTPStatus.NOT_FOUND, "Missing blueprint id")
                return
            try:
                self._send_json({"ok": True, "records": list_bindable_roles(blueprint_id)})
            except KeyError:
                self.send_error(HTTPStatus.NOT_FOUND, "Blueprint not found")
            return
        if parsed.path.startswith("/api/catalog/blueprints/"):
            blueprint_id = unquote(parsed.path.split("/api/catalog/blueprints/", 1)[1]).strip("/")
            if not blueprint_id:
                self.send_error(HTTPStatus.NOT_FOUND, "Missing blueprint id")
                return
            try:
                self._send_json({"ok": True, **get_blueprint_bundle(blueprint_id)})
            except KeyError:
                self.send_error(HTTPStatus.NOT_FOUND, "Blueprint not found")
            return
        if parsed.path.startswith("/api/results/"):
            rel_name = unquote(parsed.path.split("/api/results/", 1)[1]).strip("/")
            if not rel_name:
                self.send_error(HTTPStatus.NOT_FOUND, "Missing result filename")
                return
            file_path = (OUTPUT_RESULTS / rel_name).resolve()
            if OUTPUT_RESULTS not in file_path.parents or not file_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "Result file not found")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(file_path.stat().st_size))
            self.end_headers()
            self.wfile.write(file_path.read_bytes())
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/run":
            self._handle_run()
            return
        if parsed.path == "/api/model-preview":
            self._handle_model_preview()
            return
        if parsed.path == "/api/compose-blueprint-workload":
            self._handle_compose_blueprint_workload()
            return
        if parsed.path == "/api/compose-template-workload":
            self._handle_compose_template_workload()
            return
        if parsed.path == "/api/template-model-compatibility":
            self._handle_template_model_compatibility()
            return
        if parsed.path == "/api/blueprint-model-compatibility":
            self._handle_blueprint_model_compatibility()
            return
        if parsed.path == "/api/export-report":
            self._handle_export_report()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def _handle_run(self) -> None:
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))
            payload = json.loads(body.decode("utf-8") or "{}")
            yaml_text = str(payload.get("yaml") or "")
            if not yaml_text.strip():
                self._send_json({"ok": False, "error": "Missing generated YAML."}, status=HTTPStatus.BAD_REQUEST)
                return

            OUTPUT_SCENARIOS.mkdir(parents=True, exist_ok=True)
            OUTPUT_RESULTS.mkdir(parents=True, exist_ok=True)

            scenario_name = _extract_yaml_string_field(yaml_text, "scenario_name") or "scenario"
            scenario_stub = slugify(scenario_name, 40)
            temp_yaml_path = OUTPUT_SCENARIOS / f"{scenario_stub}_latest.yaml"
            temp_yaml_path.write_text(yaml_text, encoding="utf-8")

            parsed_yaml = yaml.safe_load(yaml_text)
            if is_workload_simulation_payload(parsed_yaml):
                scenario_base = slugify(str(parsed_yaml.get("name") or parsed_yaml.get("workload_definition", {}).get("workload_id") or "workload"), 70)
                scenario_path = OUTPUT_SCENARIOS / f"{scenario_base}.yaml"
                scenario_path.write_text(yaml_text, encoding="utf-8")
                if scenario_path != temp_yaml_path and temp_yaml_path.exists():
                    temp_yaml_path.unlink(missing_ok=True)
                result = WorkloadSimulationEngine(parsed_yaml).run()
            else:
                config = load_scenario(str(temp_yaml_path))
                scenario_base = build_base_name_from_config(config)
                scenario_path = OUTPUT_SCENARIOS / f"{scenario_base}.yaml"
                scenario_path.write_text(yaml_text, encoding="utf-8")
                if scenario_path != temp_yaml_path and temp_yaml_path.exists():
                    temp_yaml_path.unlink(missing_ok=True)
                result = run_single(config)

            result_file = OUTPUT_RESULTS / f"{scenario_base}.json"
            result_file.write_text(results_to_json([result]) + "\n", encoding="utf-8")

            self._send_json(
                {
                    "ok": True,
                    "scenario_file": str(scenario_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "result_file": str(result_file.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "result_download_url": f"/api/results/{result_file.name}",
                    "results": [result_to_dict(result)],
                    "summary": _build_summary_payload(result),
                }
            )
        except ValueError as exc:
            self._send_json({
                "ok": False,
                "error": str(exc),
            }, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - integration path
            self._send_json(
                {
                    "ok": False,
                    "error": str(exc),
                    "trace": traceback.format_exc(),
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _handle_compose_blueprint_workload(self) -> None:
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))
            payload = json.loads(body.decode("utf-8") or "{}")
            blueprint_id = str(payload.get("blueprint_id") or "").strip()
            template_id = str(payload.get("template_id") or "").strip()
            scenario_name = str(payload.get("scenario_name") or "blueprint_workload").strip()
            hardware_catalog_id = str(payload.get("hardware_catalog_id") or "").strip()
            software_stack_id = str(payload.get("software_stack_id") or "").strip()
            deployment_profile_id = str(payload.get("deployment_profile_id") or "").strip()
            if not all([blueprint_id, template_id, hardware_catalog_id, software_stack_id, deployment_profile_id]):
                self._send_json({"ok": False, "error": "Missing blueprint/template/hardware/stack/deployment selection."}, status=HTTPStatus.BAD_REQUEST)
                return
            model_overrides = {str(k): str(v) for k, v in (payload.get("model_overrides") or {}).items() if str(v).strip()}
            spec = build_workload_from_blueprint(
                blueprint_id=blueprint_id,
                template_id=template_id,
                scenario_name=scenario_name,
                hardware_catalog_id=hardware_catalog_id,
                software_stack_id=software_stack_id,
                deployment_profile_id=deployment_profile_id,
                language_code=str(payload.get("language_code") or "hu"),
                language_share=float(payload.get("language_share") or 0.8),
                model_overrides=model_overrides,
                mean_arrival_rate_per_sec=float(payload.get("mean_arrival_rate_per_sec")) if payload.get("mean_arrival_rate_per_sec") not in (None, "") else None,
                sla_latency_p95_ms=float(payload.get("sla_latency_p95_ms")) if payload.get("sla_latency_p95_ms") not in (None, "") else None,
                monte_carlo_trials=int(payload.get("monte_carlo_trials") or 40),
                time_horizon_sec=int(payload.get("time_horizon_sec") or 300),
                planning_profile=str(payload.get("planning_profile") or "baseline"),
                template_inputs={str(k): v for k, v in (payload.get("template_inputs") or {}).items()},
            )
            yaml_text = yaml.safe_dump(spec, sort_keys=False, allow_unicode=True)
            self._send_json({"ok": True, "yaml": yaml_text, "model_compatibility": spec.get("metadata", {}).get("model_compatibility") or {}})
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc), "trace": traceback.format_exc()}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


    def _handle_compose_template_workload(self) -> None:
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))
            payload = json.loads(body.decode("utf-8") or "{}")
            template_id = str(payload.get("template_id") or "").strip()
            scenario_name = str(payload.get("scenario_name") or "template_workload").strip()
            hardware_catalog_id = str(payload.get("hardware_catalog_id") or "").strip()
            software_stack_id = str(payload.get("software_stack_id") or "").strip()
            deployment_profile_id = str(payload.get("deployment_profile_id") or "").strip()
            if not all([template_id, hardware_catalog_id, software_stack_id, deployment_profile_id]):
                self._send_json({"ok": False, "error": "Missing template/hardware/stack/deployment selection."}, status=HTTPStatus.BAD_REQUEST)
                return
            model_overrides = {str(k): str(v) for k, v in (payload.get("model_overrides") or {}).items() if str(v).strip()}
            spec = build_workload_from_template(
                template_id=template_id,
                scenario_name=scenario_name,
                hardware_catalog_id=hardware_catalog_id,
                software_stack_id=software_stack_id,
                deployment_profile_id=deployment_profile_id,
                language_code=str(payload.get("language_code") or "hu"),
                language_share=float(payload.get("language_share") or 0.8),
                model_overrides=model_overrides,
                monte_carlo_trials=int(payload.get("monte_carlo_trials") or 40),
                time_horizon_sec=int(payload.get("time_horizon_sec") or 300),
                planning_profile=str(payload.get("planning_profile") or "baseline"),
                template_inputs={str(k): v for k, v in (payload.get("template_inputs") or {}).items()},
                sla_latency_p95_ms=float(payload.get("sla_latency_p95_ms")) if payload.get("sla_latency_p95_ms") not in (None, "") else None,
            )
            yaml_text = yaml.safe_dump(spec, sort_keys=False, allow_unicode=True)
            self._send_json({"ok": True, "yaml": yaml_text, "model_compatibility": spec.get("metadata", {}).get("model_compatibility") or {}})
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc), "trace": traceback.format_exc()}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


    def _handle_template_model_compatibility(self) -> None:
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))
            payload = json.loads(body.decode("utf-8") or "{}")
            role_ids = [str(x).strip() for x in (payload.get("role_ids") or []) if str(x).strip()]
            hardware_catalog_id = str(payload.get("hardware_catalog_id") or "").strip()
            software_stack_id = str(payload.get("software_stack_id") or "").strip()
            deployment_profile_id = str(payload.get("deployment_profile_id") or "").strip()
            if not all([hardware_catalog_id, software_stack_id, deployment_profile_id]):
                self._send_json({"ok": False, "error": "Missing hardware/stack/deployment selection."}, status=HTTPStatus.BAD_REQUEST)
                return
            template_family = str(payload.get("template_family") or "").strip() or None
            template_id = str(payload.get("template_id") or "").strip() or None
            if template_family is None and template_id:
                try:
                    template_record = get_advanced_workload_template_record(template_id)
                    template_family = str(template_record.get("template_family") or template_record.get("workload_family") or "").strip() or None
                except Exception:
                    template_family = None
            template_inputs = {str(k): v for k, v in (payload.get("template_inputs") or {}).items()}
            template_inputs.setdefault("_language_code", str(payload.get("language_code") or "hu"))
            template_inputs.setdefault("_language_share", float(payload.get("language_share") or 0.8))
            compatibility = assess_template_model_compatibility(
                role_model_map=build_role_model_map(role_ids, {str(k): str(v) for k, v in (payload.get("model_overrides") or {}).items() if str(v).strip()}),
                hardware_catalog_id=hardware_catalog_id,
                software_stack_id=software_stack_id,
                deployment_profile_id=deployment_profile_id,
                template_family=template_family,
                template_inputs=template_inputs,
            )
            self._send_json({"ok": True, "model_compatibility": compatibility})
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc), "trace": traceback.format_exc()}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_blueprint_model_compatibility(self) -> None:
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))
            payload = json.loads(body.decode("utf-8") or "{}")
            role_ids = [str(x).strip() for x in (payload.get("role_ids") or []) if str(x).strip()]
            hardware_catalog_id = str(payload.get("hardware_catalog_id") or "").strip()
            software_stack_id = str(payload.get("software_stack_id") or "").strip()
            deployment_profile_id = str(payload.get("deployment_profile_id") or "").strip()
            if not all([hardware_catalog_id, software_stack_id, deployment_profile_id]):
                self._send_json({"ok": False, "error": "Missing hardware/stack/deployment selection."}, status=HTTPStatus.BAD_REQUEST)
                return
            workload_family = str(payload.get("workload_family") or "").strip() or None
            if workload_family is None:
                blueprint_id = str(payload.get("blueprint_id") or "").strip() or None
                template_id = str(payload.get("template_id") or "").strip() or None
                if blueprint_id and template_id:
                    try:
                        bundle = get_blueprint_bundle(blueprint_id)
                        template = next((t for t in bundle["templates"] if str(t.get("template_id")) == template_id), None)
                        if template:
                            workload_family = str(template.get("workload_family") or "").strip() or None
                    except Exception:
                        workload_family = None
            template_inputs = {str(k): v for k, v in (payload.get("template_inputs") or {}).items()}
            template_inputs.setdefault("_language_code", str(payload.get("language_code") or "hu"))
            template_inputs.setdefault("_language_share", float(payload.get("language_share") or 0.8))
            compatibility = assess_blueprint_model_compatibility(
                role_model_map=build_role_model_map(role_ids, {str(k): str(v) for k, v in (payload.get("model_overrides") or {}).items() if str(v).strip()}),
                hardware_catalog_id=hardware_catalog_id,
                software_stack_id=software_stack_id,
                deployment_profile_id=deployment_profile_id,
                workload_family=workload_family,
                template_inputs=template_inputs,
            )
            self._send_json({"ok": True, "model_compatibility": compatibility})
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc), "trace": traceback.format_exc()}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


    def _handle_export_report(self) -> None:
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))
            payload = json.loads(body.decode("utf-8") or "{}")
            result_file = str(payload.get("result_file") or "").strip()
            export_format = str(payload.get("format") or "html").strip().lower()
            results = payload.get("results")
            if results is None:
                if not result_file:
                    self._send_json({"ok": False, "error": "Missing report input: result_file or results."}, status=HTTPStatus.BAD_REQUEST)
                    return
                result_path = (REPO_ROOT / result_file).resolve() if not result_file.startswith("/") else Path(result_file).resolve()
                if OUTPUT_RESULTS not in result_path.parents or not result_path.is_file():
                    self._send_json({"ok": False, "error": "The specified result file was not found."}, status=HTTPStatus.BAD_REQUEST)
                    return
                results = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(results, list):
                results = [results]
            report_base = slugify(str(payload.get("report_name") or (results[0].get("scenario_name") if results and isinstance(results[0], dict) else "report")), 70)
            if export_format == "pdf":
                output_path = OUTPUT_RESULTS / f"{report_base}_report.pdf"
                export_customer_report_pdf(results, output_path)
            else:
                output_path = OUTPUT_RESULTS / f"{report_base}_report.html"
                export_customer_report_html(results, output_path)
            self._send_json({
                "ok": True,
                "report_file": str(output_path.relative_to(REPO_ROOT)).replace('\\', '/'),
                "report_download_url": f"/{str(output_path.relative_to(REPO_ROOT)).replace('\\', '/')}",
                "format": export_format,
            })
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc), "trace": traceback.format_exc()}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_model_preview(self) -> None:
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))
            payload = json.loads(body.decode("utf-8") or "{}")
            model_id = str(payload.get("model_id") or "").strip()
            if not model_id:
                self._send_json({"ok": False, "error": "Missing Hugging Face model_id."}, status=HTTPStatus.BAD_REQUEST)
                return

            revision = str(payload.get("revision") or "").strip() or None
            local_files_only = bool(payload.get("local_files_only", False))
            cfg = load_model_config_from_huggingface(
                model_id,
                revision=revision,
                local_files_only=local_files_only,
            )
            self._send_json({
                "ok": True,
                "model": {
                    "name": cfg.name,
                    "total_params_billions": cfg.total_params_billions,
                    "active_params_billions": cfg.active_params_billions,
                    "num_hidden_layers": cfg.num_hidden_layers,
                    "hidden_size": cfg.hidden_size,
                    "head_dim": cfg.head_dim,
                    "num_attention_heads": cfg.num_attention_heads,
                    "num_key_value_heads": cfg.num_key_value_heads,
                    "intermediate_size": cfg.intermediate_size,
                    "moe_intermediate_size": cfg.moe_intermediate_size,
                    "num_experts": cfg.num_experts,
                    "num_experts_per_tok": cfg.num_experts_per_tok,
                    "vocab_size": cfg.vocab_size,
                    "max_position_embeddings": cfg.max_position_embeddings,
                    "is_moe": cfg.is_moe,
                },
            })
        except ValueError as exc:
            self._send_json({
                "ok": False,
                "error": str(exc),
            }, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - integration path
            self._send_json(
                {
                    "ok": False,
                    "error": str(exc),
                    "trace": traceback.format_exc(),
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _extract_yaml_string_field(yaml_text: str, field_name: str) -> str | None:
    prefix = f"{field_name}:"
    for line in yaml_text.splitlines():
        if line.strip().startswith(prefix):
            value = line.split(":", 1)[1].strip().strip('"').strip("'")
            return value or None
    return None




def _build_summary_payload(result: object) -> dict[str, object]:
    if hasattr(result, "estimated_oom_probability"):
        return {
            "risk": getattr(result, "qualitative_risk_level", "UNKNOWN"),
            "oom_probability": getattr(result, "estimated_oom_probability", None),
            "max_concurrency": getattr(result, "recommended_max_concurrency", None),
        }
    return {
        "risk": getattr(result, "qualitative_risk_level", "UNKNOWN"),
        "throughput_per_sec_mean": getattr(result, "throughput_per_sec_mean", None),
        "latency_p95_ms": getattr(result, "latency_p95_ms", None),
        "sla_hit_rate_mean": getattr(result, "sla_hit_rate_mean", None),
        "safe_capacity_24x7_per_sec": getattr(result, "safe_capacity_24x7_per_sec", None),
        "procurement_band": getattr(result, "procurement_band", None),
    }
def main() -> None:
    host = os.getenv("UI_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("UI_SERVER_PORT", "8080")))
    httpd = ThreadingHTTPServer((host, port), UIServerHandler)
    display_host = "localhost" if host == "0.0.0.0" else host
    print(f"UI server starting in container: {host}:{port}")
    print(f"Open this in your browser: http://{display_host}:{port}")
    print("Health endpoint: /api/health")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
