import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import threading
import urllib.request
from unittest import TestCase

from tests.http_test_utils import wait_for_http_server
from src.catalog_loader import get_ui_help_record, list_ui_help_records
from ui_server import UIServerHandler, ThreadingHTTPServer


class TestUiHelpCatalog(TestCase):
    def test_ui_help_catalog_loads_records(self):
        records = list_ui_help_records()
        self.assertGreaterEqual(len(records), 10)
        self.assertTrue(any(r["field_id"] == "blueprint_id" for r in records))



    def test_ui_help_catalog_has_no_dataandai_source_urls(self):
        records = list_ui_help_records()
        self.assertFalse(any('dataandai.github.io' in str((r.get('source_url') or '')) for r in records))

    def test_ui_html_tooltip_default_is_not_dataandai(self):
        html = Path('ui.html').read_text(encoding='utf-8')
        self.assertNotIn("Projekt módszertani oldal", html)
        self.assertNotIn("https://dataandai.github.io/", html)

    def test_ui_help_record_contains_source_url(self):
        record = get_ui_help_record("blueprint_id")
        self.assertIn("definition_hu", record)
        self.assertTrue(str(record.get("source_url") or "").startswith("https://"))
        self.assertIn("docs.nvidia.com", record["source_url"])


class TestUiHelpEndpoint(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), UIServerHandler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        wait_for_http_server("127.0.0.1", cls.port)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=2)

    def test_ui_help_endpoint_returns_tooltips(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/ui-help") as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        self.assertTrue(payload["ok"])
        records = payload["records"]
        self.assertTrue(any(r["field_id"] == "sla_latency_p95_ms" for r in records))
        record = next(r for r in records if r["field_id"] == "sla_latency_p95_ms")
        self.assertIn("source_url", record)
        self.assertTrue(record["source_url"].startswith("https://"))
