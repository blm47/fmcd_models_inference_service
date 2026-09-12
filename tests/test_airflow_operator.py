"""Контракт оператора с сервисом без установки Airflow и его metadata DB."""

import importlib.util
import io
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
    """Изолированный импорт: заглушка не подменяет Airflow у остальных тестов."""
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
                self.operator(partition_by="shard_id", n_shards=3).execute(self.context)
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
        operator = self.operator(partition_by="shard", n_shards=2)
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
        operator = self.operator(partition_by="shard", n_shards=3)
        with self.assertRaises(AirflowRescheduleException):
            operator.execute(self.context)
        tasks = list(self.tasks.values())
        tasks[0].update(status="FAILED", error="GPU error")
        with self.assertRaisesRegex(AirflowException, "GPU error"):
            self.operator(partition_by="shard", n_shards=3).execute(self.context)
        self.assertEqual(set(self.aborted), {task["task_id"] for task in tasks[1:]})

    def test_manual_new_attempt_uses_new_keys(self):
        first = self.operator()._requests(self.context)
        self.context["ti"].try_number += 1
        second = self.operator()._requests(self.context)
        self.assertNotEqual(first[0]["idempotency_key"], second[0]["idempotency_key"])
        self.assertEqual(first[0]["s3_output_path"], "s3://output/data")

    def test_partial_submission_failure_still_cancels_other_shards(self):
        operator = self.operator(partition_by="shard", n_shards=3)

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
            {"partition_by": "shard"},
            {"n_shards": 3},
            {"partition_by": "shard", "n_shards": 0},
            {"partition_by": "shard", "n_shards": 1.5},
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
        operator = self.operator()
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
