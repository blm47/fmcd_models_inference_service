"""Проверки единственного источника настроек и обязательных ENV."""

import copy
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from fake_s3 import config

from app.core.config import load_settings

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
    def test_yaml_is_not_overridden_by_env(self):
        with patch.dict(
            os.environ, {**ENVIRONMENT, "TASK_STORE_POLL_INTERVAL_SEC": "999"}, clear=True
        ):
            settings = load_settings()
        self.assertEqual(settings.task_store.poll_interval_sec, 3)
        self.assertEqual(settings.models[0].id_cols, ["customer_mdm_id", "partition_report_dt"])
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
            config(heartbeat_timeout_sec=1)

    def test_old_backend_config_is_rejected(self):
        with self.assertRaises(TypeError):
            config(lock_key="old.lock")
