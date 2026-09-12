"""Проверка аргументов и двух запусков Spark без подключения к Hadoop."""

import base64
import datetime as dt
import importlib.util
import inspect
import json
import sys
import types
import unittest
import zlib
from pathlib import Path
from unittest.mock import Mock, patch

from logger_stub import make_logger


class SparkSubmitStub:
    template_fields = ("_files", "_conf")

    def __init__(self, **kwargs):
        self._files = kwargs.get("files")
        self._conf = kwargs.get("conf")
        self.log = make_logger()
        self.jobs = []

    def execute(self, context):
        encoded = self._application_args[0].split("=", 1)[1].strip("'")
        self.jobs.append(json.loads(zlib.decompress(base64.b64decode(encoded))))


def load_operator():
    names = (
        "airflow",
        "airflow.hooks",
        "airflow.hooks.base",
        "airflow.providers",
        "airflow.providers.apache",
        "airflow.providers.apache.spark",
        "airflow.providers.apache.spark.operators",
        "airflow.providers.apache.spark.operators.spark_submit",
        "general_utils",
        "airflow.utils",
    )
    modules = {name: types.ModuleType(name) for name in names}
    modules["airflow.hooks.base"].BaseHook = object
    modules["airflow.utils"].timezone = types.SimpleNamespace(
        utcnow=lambda: dt.datetime.now(dt.UTC)
    )
    modules[
        "airflow.providers.apache.spark.operators.spark_submit"
    ].SparkSubmitOperator = SparkSubmitStub
    modules["general_utils"].application_args_encoding = Mock(
        side_effect=lambda payload: base64.b64encode(
            zlib.compress(json.dumps(payload).encode("utf-8"))
        ).decode("ascii")
    )
    path = Path(__file__).resolve().parents[1] / "airflow_operators" / "hadoop_to_s3_operator.py"
    spec = importlib.util.spec_from_file_location("hadoop_operator_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules), patch.object(sys, "path", sys.path.copy()):
        spec.loader.exec_module(module)
    return module.HadoopToS3Operator


Operator = load_operator()


class HadoopOperatorTests(unittest.TestCase):
    def test_sharding_parameters_reach_both_jobs(self):
        operator = self.operator(
            shard_column="customer_mdm_id",
            num_shards=9,
            shard_id_column="customer_shard",
            repartition=36,
        )
        operator.execute(self.context)
        self.assertEqual(len(operator.jobs), 2)
        for job in operator.jobs:
            self.assertEqual(job["shard_column"], "customer_mdm_id")
            self.assertEqual(job["num_shards"], 9)
            self.assertEqual(job["shard_id_column"], "customer_shard")
            self.assertEqual(job["repartition"], 36)
        for field in ("_shard_column", "_num_shards", "_shard_id_column", "_repartition"):
            self.assertIn(field, Operator.template_fields)

    def setUp(self):
        self.context = {"ti": Mock()}

    def operator(self, **kwargs):
        kwargs.setdefault("count_loaded_keys", False)
        return Operator(
            query_path="queries/data.sql",
            endpoint="https://s3.example",
            s3_bucket="data",
            access_key="test",
            secret_key="test",
            **kwargs,
        )

    def test_only_correct_metrics_argument(self):
        params = inspect.signature(Operator).parameters
        self.assertIn("metrics_calc_variables", params)
        self.assertEqual(
            [name for name in params if "calc_variables" in name], ["metrics_calc_variables"]
        )

    def test_none_and_empty_dict_disable_metadata(self):
        for metrics in (None, {}):
            with self.subTest(metrics=metrics):
                operator = self.operator(metrics_calc_variables=metrics)
                operator.execute(self.context)
                self.assertEqual(len(operator.jobs), 1)

    def test_default_metadata_never_clears_data(self):
        operator = self.operator(clear_s3_path=True)
        operator.execute(self.context)
        self.assertEqual(len(operator.jobs), 2)
        self.assertTrue(operator.jobs[0]["clear_s3_path"])
        self.assertFalse(operator.jobs[1]["clear_s3_path"])
        self.assertTrue(Path(operator._application).is_file())
        self.assertEqual(Path(operator._application).parent.name, "scripts")

    def test_uses_library_encoder(self):
        operator = self.operator(metrics_calc_variables=None)
        encoder = Mock(return_value="encoded-by-library")
        with patch.dict(Operator._run_spark_job.__globals__, application_args_encoding=encoder):
            with patch.object(SparkSubmitStub, "execute"):
                operator.execute(self.context)
        self.assertEqual(encoder.call_count, 2)
        self.assertTrue(encoder.call_args_list[0].args[0]["use_bulk_committer"])
        self.assertEqual(
            encoder.call_args_list[1].args[0], {"access_key": "test", "secret_key": "test"}
        )
        self.assertEqual(
            operator._application_args, ["--application_args_encoded='encoded-by-library'"]
        )

    def test_metrics_are_copied_per_task(self):
        metrics = {"load_id": "source"}
        first = self.operator(metrics_calc_variables=metrics)
        second = self.operator(metrics_calc_variables=metrics)
        first._meta_variables["load_id"] = "changed"
        self.assertEqual(second._meta_variables, metrics)
        first = self.operator()
        second = self.operator()
        first._meta_variables.clear()
        self.assertTrue(second._meta_variables)

    def test_latest_metadata_settings_and_mark_loaded_at(self):
        operator = self.operator(no_delete_meta_file=True, dt_formatter="%Y-%m-%d", load_id="fixed")
        operator.execute(self.context)
        self.context["ti"].xcom_push.assert_called_once()
        self.assertEqual(self.context["ti"].xcom_push.call_args.kwargs["key"], "mark_loaded_at")
        self.assertEqual(operator.jobs[0]["load_id"], "fixed")
        self.assertEqual(operator.jobs[0]["dt_formatter"], "%Y-%m-%d")
        self.assertEqual(
            operator.jobs[0]["load_start_time"], dt.datetime.now().strftime("%Y-%m-%d")
        )
        self.assertTrue(operator.jobs[1]["no_delete_meta_file"])
        self.assertIn("status_sumk", operator.jobs[1]["meta_variables"])

    def test_count_current_load_parquet_across_pages(self):
        client = Mock()
        client.get_paginator.return_value.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "data/load_id=current/shard_id=0/part-0.parquet"},
                    {"Key": "data/load_id=current/_SUCCESS"},
                ]
            },
            {},
            {"Contents": [{"Key": "data/load_id=current/shard_id=1/part-1.parquet"}]},
        ]
        operator = self.operator(s3_path="data", load_id="current", count_loaded_keys=True)
        with patch("boto3.client", return_value=client):
            operator.execute(self.context)
        client.get_paginator.return_value.paginate.assert_called_once_with(
            Bucket="data", Prefix="data/load_id=current/"
        )
        self.context["ti"].xcom_push.assert_any_call(key="cnt_loaded_keys", value=2)
        client.close.assert_called_once()

    def test_count_failure_is_optional(self):
        operator = self.operator(count_loaded_keys=True)
        with patch("boto3.client", side_effect=RuntimeError("S3 недоступен")):
            operator.execute(self.context)
        self.context["ti"].xcom_push.assert_any_call(key="cnt_loaded_keys", value=-1)

    def test_disabled_count_does_not_connect_to_s3(self):
        with patch("boto3.client") as client:
            self.operator().execute(self.context)
        client.assert_not_called()

    def test_count_without_load_id_handles_rendered_false(self):
        client = Mock()
        client.get_paginator.return_value.paginate.return_value = []
        operator = self.operator(s3_path="data/", add_load_id="False", count_loaded_keys=True)
        with patch("boto3.client", return_value=client):
            operator.execute(self.context)
        client.get_paginator.return_value.paginate.assert_called_once_with(
            Bucket="data", Prefix="data/"
        )
        self.context["ti"].xcom_push.assert_any_call(key="cnt_loaded_keys", value=0)
