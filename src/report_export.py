from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .report_payload import build_customer_report_payload


TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


def render_customer_report_html(results: list[Any]) -> str:
    payload = build_customer_report_payload(results)
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(enabled_extensions=("html", "xml")),
    )
    template = env.get_template("report_customer_offer.html.j2")
    return template.render(report=payload)



def export_customer_report_html(results: list[Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_customer_report_html(results), encoding="utf-8")
    return output
