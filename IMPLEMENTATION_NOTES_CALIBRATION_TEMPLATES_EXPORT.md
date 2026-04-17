# Implementation notes - calibration, advanced templates, report export

## Added components

### Performance calibration
- `catalog/calibration/benchmark_observations.json`
- `catalog/calibration/calibration_anchors.json`
- `catalog/calibration/calibration_anchors.schema.json`
- `src/calibration_loader.py`
- `src/calibration_rules.py`

The workload engine now resolves stage-level benchmark anchors from blueprint/template/workload family, runtime, deployment, and hardware context, then applies calibrated stage service-time baselines before the existing runtime/deployment penalties.

### Advanced workload template library
- `catalog/workload_templates/advanced_workload_templates.json`
- `src/template_adapter.py`
- `examples/workload_simulation/advanced_templates/*.yaml`

Supported template seeds:
- Retrieval + Rerank + Generation
- Multi-stage Agentic Workflow
- Batch Document Pipeline
- Streaming / Queue-heavy Inference
- Hybrid Multimodal Pipeline

### Report export
- `src/report_payload.py`
- `src/report_export.py`
- `src/pdf_export.py`
- `templates/report_customer_offer.html.j2`

Export paths:
- CLI: `run_all.py ... --report-html` / `--report-pdf`
- Existing result JSON: `run_all.py --report-from-json output/results/<file>.json --report-pdf`
- UI server: `POST /api/export-report`

## UI additions
- New `Advanced Templates` tab in `ui.html`
- New API endpoints:
  - `GET /api/catalog/workload-templates`
  - `POST /api/compose-template-workload`
  - `POST /api/export-report`

## Audit / traceability
- Workload simulation results now include `calibration_trace`
- Calibration coverage is included in result notes
- Report appendix carries calibration and detailed stage traces

## Validation status
- test suite passes: `61 passed`
