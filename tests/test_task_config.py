import copy
import os
import unittest
from unittest.mock import patch

from app.core import config


class TaskStoreConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.raw = config._load_yaml("configs/models.yaml")
        self.environment = {
            "S3_ENDPOINT_URL": "https://s3.example.invalid",
            "S3_ACCESS_KEY": "test-key",
            "S3_SECRET_KEY": "test-secret",
            "S3_BUCKET_IN": "input",
            "S3_BUCKET_OUT": "output",
        }
        self.cache = patch.object(config, "_settings_cache", None)
        self.cache.start()
        self.addCleanup(self.cache.stop)

    def load(self, raw=None, environment=None):
        with (
            patch.dict(os.environ, environment or self.environment, clear=True),
            patch.object(config, "_load_yaml", return_value=raw or self.raw),
        ):
            return config.load_settings()

    def test_missing_task_store_section_has_usable_defaults(self):
        raw = copy.deepcopy(self.raw)
        raw.pop("task_store")
        settings = self.load(raw)
        self.assertEqual(settings.task_store, config.TaskStoreConfig())
        self.assertNotEqual(settings.task_store.state_key, settings.task_store.lock_key)

    def test_environment_overrides_yaml_liveness_and_retention(self):
        environment = {
            **self.environment,
            "TASK_STORE_HEARTBEAT_INTERVAL_SEC": "15.5",
            "TASK_STORE_HEARTBEAT_TIMEOUT_SEC": "90",
            "TASK_STORE_CLEANUP_INTERVAL_SEC": "120",
            "TASK_STORE_RETENTION_MONTHS": "2",
            "TASK_STORE_STATE_KEY": "custom/state.json",
            "TASK_STORE_LOCK_KEY": "custom/state.lock",
            "TASK_STORE_LEASE_SECONDS": "70",
            "TASK_STORE_WAIT_TIMEOUT_SEC": "310",
            "TASK_STORE_POLL_INTERVAL_SEC": "0.5",
        }
        settings = self.load(environment=environment)
        self.assertEqual(settings.task_store.heartbeat_interval_sec, 15.5)
        self.assertEqual(settings.task_store.heartbeat_timeout_sec, 90)
        self.assertEqual(settings.task_store.cleanup_interval_sec, 120)
        self.assertEqual(settings.task_store.retention_months, 2)
        self.assertEqual(settings.task_store.state_key, "custom/state.json")
        self.assertEqual(settings.task_store.lock_key, "custom/state.lock")
        self.assertEqual(settings.task_store.lease_seconds, 70)
        self.assertEqual(settings.task_store.wait_timeout_sec, 310)
        self.assertEqual(settings.task_store.poll_interval_sec, 0.5)

    def test_yaml_liveness_fields_are_loaded(self):
        raw = copy.deepcopy(self.raw)
        raw["task_store"].update(
            heartbeat_interval_sec=45,
            heartbeat_timeout_sec=240,
            cleanup_interval_sec=7200,
        )
        settings = self.load(raw)
        self.assertEqual(settings.task_store.heartbeat_interval_sec, 45)
        self.assertEqual(settings.task_store.heartbeat_timeout_sec, 240)
        self.assertEqual(settings.task_store.cleanup_interval_sec, 7200)

    def test_invalid_timing_rejected_before_startup(self):
        for name in (
            "lease_seconds", "wait_timeout_sec", "poll_interval_sec",
            "heartbeat_interval_sec", "heartbeat_timeout_sec", "cleanup_interval_sec",
        ):
            for value in (0, -1, float("inf"), float("nan")):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        config.TaskStoreConfig(**{name: value})

    def test_timeout_must_exceed_heartbeat_period(self):
        with self.assertRaisesRegex(ValueError, "heartbeat_timeout_sec"):
            config.TaskStoreConfig(heartbeat_interval_sec=30, heartbeat_timeout_sec=30)

    def test_retention_requires_positive_whole_months(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "retention_months"):
                    config.TaskStoreConfig(retention_months=value)

    def test_state_and_lock_cannot_overwrite_each_other(self):
        with self.assertRaisesRegex(ValueError, "must be different"):
            config.TaskStoreConfig(state_key="/same", lock_key="same")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            config.TaskStoreConfig(state_key="/")


if __name__ == "__main__":
    unittest.main()
