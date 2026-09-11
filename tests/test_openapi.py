"""Документация доступна без обращения к S3 и загрузки GPU-моделей."""

import unittest
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from app.main import app


class OpenApiTests(unittest.TestCase):
    def test_exported_schema_is_current(self):
        exported_path = Path(__file__).resolve().parents[1] / "docs" / "openapi.yaml"
        exported = yaml.safe_load(exported_path.read_text(encoding="utf-8"))
        self.assertEqual(exported, app.openapi())

    def test_documentation_endpoints(self):
        client = TestClient(app)
        self.addCleanup(client.close)
        for path in ("/docs", "/redoc", "/openapi.json"):
            with self.subTest(path=path):
                self.assertEqual(client.get(path).status_code, 200)
        schema = client.get("/openapi.json").json()
        infer = schema["paths"]["/infer"]["post"]
        self.assertEqual(set(infer["responses"]), {"202", "404", "409", "422", "503"})
        request = schema["components"]["schemas"]["InferRequest"]
        self.assertIn("idempotency_key", request["required"])
        self.assertIn("examples", request["properties"]["s3_input_path"])
        # Проверяем все опубликованные методы, чтобы новые маршруты не оставались без описания.
        for path, operations in schema["paths"].items():
            for method, operation in operations.items():
                with self.subTest(path=path, method=method):
                    self.assertTrue(operation["summary"])
                    self.assertTrue(operation["description"])
                    self.assertTrue(operation["tags"])
                    success = "202" if path == "/infer" else "200"
                    self.assertIn(
                        "schema", operation["responses"][success]["content"]["application/json"]
                    )
