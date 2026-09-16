"""
Проверки единственного источника настроек и обязательных ENV.
"""

import copy
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from fake_s3 import config

from app.core.config import PostgresConfig, load_settings

ENVIRONMENT = {
    "S3_ENDPOINT_URL": "https://s3.invalid",
    "S3_ACCESS_KEY": "test",
    "S3_SECRET_KEY": "test",
    "S3_BUCKET_IN": "input",
    "S3_BUCKET_OUT": "output",
    "S3_REGION": "us-east-1",
    "S3_USE_SSL": "true",
    "S3_VERIFY_SSL": "false",
}


class ConfigTests(unittest.TestCase):
    def test_unknown_backend_and_old_global_settings_are_rejected(self):
        raw = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))
        for changed in (
            {**raw, "inference": {}},
            {**raw, "models": [{**raw["models"][0], "backend": "unknown"}]},
        ):
            with patch("app.core.config.yaml.safe_load", return_value=changed):
                with self.assertRaises(ValueError):
                    load_settings()

    def test_yaml_is_not_overridden_by_env(self):
        with patch.dict(
            os.environ, {**ENVIRONMENT, "TASK_STORE_POLL_INTERVAL_SEC": "999"}, clear=True
        ):
            settings = load_settings()
        self.assertEqual(settings.task_store.poll_interval_sec, 3)
        self.assertEqual(settings.models[0].id_cols, ("customer_mdm_id", "partition_report_dt"))
        self.assertFalse(settings.s3.verify_ssl)

    def test_each_s3_env_is_required(self):
        for key in ENVIRONMENT:
            env = {name: value for name, value in ENVIRONMENT.items() if name != key}
            with self.subTest(key=key), patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(ValueError, key):
                    load_settings()

    def test_boolean_typo_is_rejected(self):
        with patch.dict(os.environ, {**ENVIRONMENT, "S3_VERIFY_SSL": "treu"}, clear=True):
            with self.assertRaisesRegex(ValueError, "S3_VERIFY_SSL"):
                load_settings()

    def test_missing_yaml_value_has_no_code_default(self):
        raw = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))
        changed = copy.deepcopy(raw)
        del changed["task_store"]["poll_interval_sec"]
        with patch("app.core.config.yaml.safe_load", return_value=changed):
            with self.assertRaises(TypeError):
                load_settings()

    def test_invalid_intervals_rejected(self):
        for value in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                config(poll_interval_sec=value)
        with self.assertRaises(ValueError):
            config(task_timeout_sec=1)
        with self.assertRaisesRegex(ValueError, "heartbeat_interval_sec"):
            config(heartbeat_interval_sec=1800)

    def test_postgres_environment_is_required_only_for_pg(self):
        raw = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))
        raw["task_store"]["backend"] = "pg"
        postgres = {
            "POSTGRES_URL": "postgresql://localhost:5432/test",
            "POSTGRES_LOGIN": "worker",
            "POSTGRES_PASSWORD": "secret",
            "POSTGRES_SCHEMA": "fmcd",
            "POSTGRES_TABLE_NAME": "tasks",
        }
        with patch("app.core.config.yaml.safe_load", return_value=raw):
            for key in postgres:
                env = {
                    **ENVIRONMENT,
                    **{name: value for name, value in postgres.items() if name != key},
                }
                with self.subTest(key=key), patch.dict(os.environ, env, clear=True):
                    with self.assertRaisesRegex(ValueError, key):
                        load_settings()
            with patch.dict(os.environ, {**ENVIRONMENT, **postgres}, clear=True):
                settings = load_settings()
                self.assertEqual(settings.task_store.backend, "pg")
                self.assertEqual(settings.postgres.table_name, "tasks")
                self.assertNotIn("secret", repr(settings.postgres))

    def test_invalid_storage_backend_and_postgres_config_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "backend"):
            config(backend="sqlite")
        for url in ("localhost", "http://host/db", "postgresql://login:password@host/db"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                PostgresConfig(url, "login", "password", "schema", "tasks")

    def test_old_backend_config_is_rejected(self):
        with self.assertRaises(TypeError):
            config(lock_key="old.lock")
