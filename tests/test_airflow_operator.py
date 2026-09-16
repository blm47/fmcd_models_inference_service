"""
Контракт оператора с сервисом без установки Airflow и его metadata DB.
"""

import importlib.util
import io
import ssl
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from logger_stub import make_logger


class AirflowException(Exception):
    pass


class AirflowRescheduleException(Exception):
    pass


class AirflowTaskTimeout(BaseException):
    pass


class PokeReturnValue:
    def __init__(self, is_done, xcom_value):
        self.is_done = is_done
        self.xcom_value = xcom_value


class BaseSensorOperator:
    def __init__(self, **kwargs):
        self.log = make_logger()
        self.options = kwargs

    def execute(self, context):
        result = self.poke(context)
        if result:
            return result.xcom_value
        raise AirflowRescheduleException()


def load_operator():
    """
    Изолированный импорт: заглушка не подменяет Airflow у остальных тестов.
    """
    modules = {
        name: types.ModuleType(name)
        for name in ("airflow", "airflow.exceptions", "airflow.sensors", "airflow.sensors.base")
    }
    modules["airflow.exceptions"].AirflowException = AirflowException
    modules["airflow.exceptions"].AirflowRescheduleException = AirflowRescheduleException
    modules["airflow.exceptions"].AirflowTaskTimeout = AirflowTaskTimeout
    modules["airflow.sensors.base"].BaseSensorOperator = BaseSensorOperator
    modules["airflow.sensors.base"].PokeReturnValue = PokeReturnValue
    path = (
        Path(__file__).resolve().parents[1]
        / "airflow_operators"
        / "gpu_model_inference_operator.py"
    )
    spec = importlib.util.spec_from_file_location("operator_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module.GPUModelInferenceOperator


Operator = load_operator()


class InferenceOperatorTests(unittest.TestCase):
    def test_utilization_flag_is_sent_for_every_shard(self):
        self.assertFalse(
            self.operator(shard_id_column=None)._requests(self.context)[0]["calc_utilization"]
        )
        operator = self.operator(calc_utilization=True, shard_id_column="shard_id", num_shards=3)
        requests = operator._requests(self.context)
        self.assertEqual(len(requests), 3)
        self.assertTrue(all(payload["calc_utilization"] is True for payload in requests))
        with self.assertRaisesRegex(ValueError, "calc_utilization"):
            self.operator(calc_utilization="False")

    def test_default_shard_column_is_used_when_count_is_provided(self):
        operator = self.operator(num_shards=2)
        self.assertEqual(operator.shard_id_column, "shard_id")
        requests = operator._requests(self.context)
        self.assertEqual(
            [payload["s3_input_path"] for payload in requests],
            [f"s3://input/data/shard_id={index}" for index in range(2)],
        )
        self.assertEqual(
            [payload["s3_output_path"] for payload in requests],
            [f"s3://output/data/shard_id={index}" for index in range(2)],
        )
        self.assertNotEqual(requests[0]["idempotency_key"], requests[1]["idempotency_key"])

    def test_explicitly_disabled_sharding_submits_one_request(self):
        operator = self.operator(shard_id_column=None)
        with self.assertRaises(AirflowRescheduleException):
            operator.execute(self.context)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0]["s3_input_path"], "s3://input/data")
        self.assertEqual(self.requests[0]["s3_output_path"], "s3://output/data")

    def test_progress_is_logged_on_every_poll_and_100_is_not_done(self):
        operator = self.operator(shard_id_column=None)
        payload = operator._requests(self.context)[0]
        key = payload["idempotency_key"]
        self.tasks[key] = {
            "task_id": key,
            "status": "RUNNING",
            "processed_rows": 25,
            "total_rows": 100,
            "progress_pct": 25.0,
            "eta_seconds": 120.0,
        }
        with self.assertRaises(AirflowRescheduleException):
            operator.execute(self.context)
        message = operator.log.info.call_args.args[0]
        self.assertIn("прогресс=25.0%", message)
        self.assertIn("строк=25/100", message)
        self.assertIn("ETA=120 с", message)
        self.tasks[key].update(processed_rows=100, progress_pct=100.0, eta_seconds=0.0)
        with self.assertRaises(AirflowRescheduleException):
            operator.execute(self.context)
        self.assertEqual(operator.log.info.call_count, 2)
        self.assertIn("ETA=0 с", operator.log.info.call_args.args[0])

    def test_unavailable_eta_and_zero_progress_are_logged(self):
        operator = self.operator(shard_id_column=None)
        key = operator._requests(self.context)[0]["idempotency_key"]
        self.tasks[key] = {
            "task_id": key,
            "status": "QUEUED",
            "processed_rows": 0,
            "total_rows": 100,
            "progress_pct": 0.0,
            "eta_seconds": None,
        }
        with self.assertRaises(AirflowRescheduleException):
            operator.execute(self.context)
        message = operator.log.info.call_args.args[0]
        self.assertIn("прогресс=0.0%", message)
        self.assertIn("ETA=нет оценки", message)

    def test_tls_verification_for_submission_status_and_abort(self):
        for verify in (True, False):
            operator = self.operator(verify_ssl=verify)
            operator.service_url = "https://inference.example"
            for method, path in (
                ("POST", "/infer"),
                ("GET", "/tasks/id/status"),
                ("POST", "/tasks/id/abort"),
            ):
                with self.subTest(verify=verify, path=path):
                    with patch.dict(
                        Operator._http.__globals__,
                        urlopen=lambda *a, verify=verify, **kw: self.tls_response(kw, verify),
                    ):
                        Operator._http(operator, method, path)
        self.assertFalse(self.operator().verify_ssl)
        with self.assertRaises(ValueError):
            self.operator(verify_ssl="False")

    def tls_response(self, kwargs, verify):
        context = kwargs["context"]
        self.assertEqual(context.check_hostname, verify)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED if verify else ssl.CERT_NONE)
        return io.BytesIO(b"{}")

    def setUp(self):
        self.context = {
            "ti": types.SimpleNamespace(
                dag_id="model", run_id="run-1", task_id="infer", map_index=-1, try_number=1
            )
        }
        self.tasks = {}
        self.requests = []
        self.aborted = []

    def operator(self, **kwargs):
        operator = Operator(
            task_id="infer",
            service_url="http://inference:8080",
            s3_input_prefix="s3://input/data/",
            s3_output_prefix="s3://output/data/",
            model_name="credit_cards",
            **kwargs,
        )
        operator._http = self.http
        return operator

    def http(self, method, path, payload=None):
        if path == "/infer":
            self.requests.append(payload)
            key = payload["idempotency_key"]
            self.tasks.setdefault(key, {"task_id": key, "status": "QUEUED"})
            return self.tasks[key].copy()
        task_id = path.split("/")[2]
        if path.endswith("/abort"):
            self.aborted.append(task_id)
            self.tasks[task_id]["status"] = "ABORTED"
        return self.tasks[task_id].copy()

    def test_reschedule_restores_same_tasks_without_xcom(self):
        for _ in range(2):
            with self.assertRaises(AirflowRescheduleException):
                self.operator(shard_id_column="shard_id", num_shards=3).execute(self.context)
        self.assertEqual(len(self.tasks), 3)
        self.assertEqual(self.aborted, [])
        self.assertEqual(
            [r["s3_input_path"] for r in self.requests[:3]],
            [f"s3://input/data/shard_id={i}" for i in range(3)],
        )
        self.assertEqual(
            [r["s3_output_path"] for r in self.requests[:3]],
            [f"s3://output/data/shard_id={i}" for i in range(3)],
        )

    def test_success_requires_all_done_and_returns_ids(self):
        operator = self.operator(shard_id_column="shard", num_shards=2)
        with self.assertRaises(AirflowRescheduleException):
            operator.execute(self.context)
        tasks = list(self.tasks.values())
        tasks[0]["status"] = "DONE"
        with self.assertRaises(AirflowRescheduleException):
            operator.execute(self.context)
        tasks[1]["status"] = "DONE"
        result = operator.execute(self.context)
        self.assertEqual(set(result["task_ids"]), set(self.tasks))
        self.assertEqual(result["s3_output_prefix"], "s3://output/data")

    def test_failure_aborts_every_other_active_shard(self):
        operator = self.operator(shard_id_column="shard", num_shards=3)
        with self.assertRaises(AirflowRescheduleException):
            operator.execute(self.context)
        tasks = list(self.tasks.values())
        tasks[0].update(status="FAILED", error="GPU error")
        with self.assertRaisesRegex(AirflowException, "GPU error"):
            self.operator(shard_id_column="shard", num_shards=3).execute(self.context)
        self.assertEqual(set(self.aborted), {task["task_id"] for task in tasks[1:]})

    def test_manual_new_attempt_uses_new_keys(self):
        first = self.operator(shard_id_column=None)._requests(self.context)
        self.context["ti"].try_number += 1
        second = self.operator(shard_id_column=None)._requests(self.context)
        self.assertNotEqual(first[0]["idempotency_key"], second[0]["idempotency_key"])
        self.assertEqual(first[0]["s3_output_path"], "s3://output/data")

    def test_partial_submission_failure_still_cancels_other_shards(self):
        operator = self.operator(shard_id_column="shard", num_shards=3)

        def http(method, path, payload=None):
            if payload and payload["s3_input_path"].endswith("=1"):
                raise AirflowException("HTTP 422")
            return self.http(method, path, payload)

        operator._http = http
        with self.assertRaisesRegex(AirflowException, "HTTP 422"):
            operator.execute(self.context)
        self.assertEqual(len(self.aborted), 2)

    def test_abort_failure_does_not_skip_other_tasks(self):
        operator = self.operator()
        operator._tasks = {"one": {"status": "FINALIZING"}, "two": {"status": "RUNNING"}}
        calls = []

        def http(method, path, payload=None):
            calls.append(path)
            if "one" in path:
                raise AirflowException("HTTP 409")
            return {"status": "ABORTING"}

        operator._http = http
        operator.on_kill()
        self.assertEqual(len(calls), 2)
        operator.log.error.assert_called_once()

    def test_invalid_shard_configuration(self):
        for kwargs in (
            {"shard_id_column": "shard"},
            {"shard_id_column": None, "num_shards": 3},
            {"shard_id_column": "shard", "num_shards": 0},
            {"shard_id_column": "shard", "num_shards": 1.5},
            {"shard_id_column": "shard", "num_shards": True},
            {"shard_id_column": "invalid/name", "num_shards": 3},
            {"shard_id_column": "", "num_shards": 3},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.operator(**kwargs)._requests(self.context)

    def test_automatic_retries_rejected(self):
        with self.assertRaises(ValueError):
            self.operator(retries=1)

    def test_task_timeout_cancels_known_tasks(self):
        operator = self.operator()

        def timeout(context):
            operator._tasks = {"task-1": {"status": "RUNNING"}}
            raise AirflowTaskTimeout()

        with patch.object(BaseSensorOperator, "execute", side_effect=timeout):
            with patch.object(operator, "_http", return_value={"status": "ABORTING"}) as http:
                with self.assertRaises(AirflowTaskTimeout):
                    operator.execute(self.context)
                http.assert_called_once_with("POST", "/tasks/task-1/abort")

    def test_lost_http_response_retries_same_payload(self):
        operator = self.operator(shard_id_column=None)
        payload = operator._requests(self.context)[0]
        globals_ = Operator._http.__globals__
        from unittest.mock import Mock

        opener = Mock(
            side_effect=[URLError("ответ потерян"), io.BytesIO(b'{"task_id": "existing-task"}')]
        )
        with patch.dict(globals_, {"urlopen": opener}), patch.object(globals_["time"], "sleep"):
            result = Operator._http(operator, "POST", "/infer", payload)
        self.assertEqual(result["task_id"], "existing-task")
        first, second = opener.call_args_list
        self.assertEqual(first.args[0].data, second.args[0].data)
