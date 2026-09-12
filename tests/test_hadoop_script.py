"""Проверки Spark driver: метаданные, транспорт и настройки без кластера Hadoop."""

import base64
import io
import json
import types
import unittest
import zlib
from unittest.mock import Mock, patch

import pandas as pd
from botocore.exceptions import ClientError
from logger_stub import make_logger

from airflow_operators.scripts import hadoop_to_s3_script as driver


def encode(payload):
    return base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()


class HadoopScriptTests(unittest.TestCase):
    def test_sharding_is_independent_of_write_parallelism(self):
        for count in (1, 36):
            args = driver.normalize_args(
                dict(
                    shard_column="customer_mdm_id",
                    num_shards="9",
                    repartition=str(count),
                )
            )
            self.assertEqual(args["partition_by"], ["shard_id"])
            self.assertEqual(args["num_shards"], 9)
            self.assertEqual(args["repartition"], count)
        self.assertEqual(
            driver.normalize_args(
                dict(
                    shard_column="customer_mdm_id",
                    num_shards=9,
                )
            )["repartition"],
            1,
        )
        self.assertEqual(driver.normalize_args({})["partition_by"], [])

    def test_invalid_sharding_configuration(self):
        for values in (
            {"num_shards": 9},
            {"shard_column": "id"},
            {"shard_column": "id", "num_shards": 0},
            {"shard_column": "id", "num_shards": True},
            {"shard_column": "id", "num_shards": 1.5},
            {"shard_column": "id", "num_shards": 9, "repartition": 0},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                driver.normalize_args(values)

    def test_query_wrapper_preserves_cte_comments_and_literals(self):
        sql = "WITH src AS (SELECT 1 AS id, ';' AS value) SELECT * FROM src; -- конец"
        wrapped = driver.wrap_query(sql, "id", 9, "bucket_id")
        self.assertIn("pmod(xxhash64(_shard_source.`id`), 9) AS `bucket_id`", wrapped)
        self.assertIn("';' AS value", wrapped)
        self.assertIn("FROM src -- конец\n) AS _shard_source", wrapped)
        self.assertIn("`odd``key`", driver.wrap_query("SELECT 1", "odd`key", 9))
        with self.assertRaises(ValueError):
            driver.wrap_query("SELECT 1; SELECT 2", "id", 9)

    def test_duplicate_output_columns_and_missing_partition_fail(self):
        args = driver.normalize_args(dict(shard_column="id", num_shards=9))
        for columns in (["value"], ["id", "SHARD_ID", "shard_id"], ["id", "ID", "shard_id"]):
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                driver.validate_columns(columns, args)

    def test_jinja_is_rendered_before_wrapping_and_writing(self):
        self.args.update(
            driver.normalize_args(
                dict(
                    shard_column="customer_mdm_id",
                    num_shards=9,
                    repartition=36,
                    query_path="query.sql",
                    template_variables={"table": "source_table"},
                )
            )
        )
        sharded = Mock(columns=["customer_mdm_id", "value", "shard_id"])
        sharded.rdd.isEmpty.return_value = False
        self.spark.sql.return_value = sharded
        self.spark.conf.get.return_value = "false"
        with (
            patch.object(driver, "read_sql_file", return_value="SELECT * FROM {{ table }};"),
            patch.object(driver, "write_data") as write,
        ):
            driver.run(self.spark, self.args, self.logger)
        wrapped = self.spark.sql.call_args.args[0]
        self.spark.sql.assert_called_once()
        self.assertIn("SELECT * FROM source_table\n)", wrapped)
        self.assertIn("pmod(xxhash64(_shard_source.`customer_mdm_id`), 9)", wrapped)
        self.assertIs(write.call_args.args[1], sharded)

    def test_metadata_never_clears_or_shards_data(self):
        self.args.update(
            driver.normalize_args(
                dict(
                    shard_column="id",
                    num_shards=9,
                    clear_s3_path=True,
                    use_bulk_committer=False,
                    query_path="query.sql",
                )
            )
        )
        frame = Mock()
        frame.rdd.isEmpty.return_value = False
        self.spark.sql.return_value = frame
        with (
            patch.object(driver, "read_sql_file", return_value="SELECT id FROM source"),
            patch.object(driver, "write_metadata") as metadata,
        ):
            driver.run(self.spark, self.args, self.logger)
        self.spark.sql.assert_called_once_with("SELECT id FROM source")
        metadata.assert_called_once()
        self.spark._jvm.org.apache.hadoop.fs.Path.assert_not_called()

    def test_schema_error_prevents_s3_cleanup(self):
        self.args.update(query_path="query.sql", clear_s3_path=True)
        self.spark.conf.get.return_value = "false"
        self.spark.sql.return_value.columns = ["id", "ID"]
        with patch.object(driver, "read_sql_file", return_value="SELECT 1 AS id, 2 AS ID"):
            with self.assertRaises(ValueError):
                driver.run(self.spark, self.args, self.logger)
        self.spark._jvm.org.apache.hadoop.fs.Path.assert_not_called()

    def test_unsharded_query_and_empty_result(self):
        self.args.update(query_path="query.sql", repartition=36, clear_s3_path=True)
        self.spark.conf.get.return_value = "false"
        frame = Mock(columns=["id"])
        self.spark.sql.return_value = frame
        for empty in (True, False):
            frame.rdd.isEmpty.return_value = empty
            with (
                patch.object(driver, "read_sql_file", return_value="SELECT 1 AS id;"),
                patch.object(driver, "write_data") as write,
            ):
                driver.run(self.spark, self.args, self.logger)
            self.spark.sql.assert_called_with("SELECT 1 AS id;")
            self.assertEqual(write.call_count, 0 if empty else 1)

    def test_bool_templates_and_partition_names(self):
        args = driver.normalize_args(
            dict(
                shard_column="id",
                num_shards="9",
                add_load_id="False",
                clear_s3_path="0",
                safe_upload="true",
                partition_by="business_date",
            )
        )
        self.assertFalse(args["add_load_id"])
        self.assertFalse(args["clear_s3_path"])
        self.assertTrue(args["safe_upload"])
        self.assertEqual(args["partition_by"], ["business_date", "shard_id"])
        driver.validate_columns(["id", "business_date", "shard_id"], args)
        with self.assertRaises(ValueError):
            driver.normalize_args(dict(mode="ignore", clear_s3_path=True))

    def setUp(self):
        self.args = driver.normalize_args(
            {
                "endpoint": "https://s3.example",
                "s3_bucket": "data",
                "s3_path": "dataset",
                "s3_meta_path": "meta/table",
                "object_name": "db.table",
                "load_id": "current",
                "load_start_time": "2026-09-12",
                "dt_formatter": "%Y-%m-%d",
                "meta_variables": {
                    "object_name": "None",
                    "inc_rowcount": "None",
                    "start_time": "None",
                    "finish_time": "None",
                },
            }
        )
        self.logger = make_logger()
        self.spark = Mock()
        self.spark.sparkContext.getConf.return_value.get.return_value = encode(
            {"access_key": "test", "secret_key": "test"}
        )

    def test_transport_preserves_quoted_library_payload(self):
        self.assertEqual(driver.decode_payload("'" + encode(self.args) + "'"), self.args)
        self.assertEqual(
            driver.normalize_args(driver.decode_payload(encode(self.args)))["load_id"], "current"
        )

    def test_boto_csv_retains_original_key_and_appends_row(self):
        client = Mock()
        body = io.BytesIO(b"object_name,inc_rowcount\nold,10\n")
        client.get_object.return_value = {"Body": body}
        metadata = pd.DataFrame({"object_name": ["new"], "inc_rowcount": [20]})
        with patch("boto3.client", return_value=client):
            driver.append_csv_boto(self.spark, self.args, metadata, self.logger)
        client.get_object.assert_called_once_with(Bucket="data", Key="metatable/table.csv")
        saved = client.put_object.call_args.kwargs
        self.assertEqual(saved["Key"], "metatable/table.csv")
        self.assertEqual(pd.read_csv(io.StringIO(saved["Body"]))["inc_rowcount"].tolist(), [10, 20])
        client.delete_object.assert_not_called()
        client.close.assert_called_once()
        self.assertTrue(body.closed)

    def test_missing_csv_is_created_but_access_denied_is_not_overwritten(self):
        for code in ("NoSuchKey", "AccessDenied"):
            with self.subTest(code=code):
                client = Mock()
                client.get_object.side_effect = ClientError({"Error": {"Code": code}}, "GetObject")
                with patch("boto3.client", return_value=client):
                    if code == "NoSuchKey":
                        driver.append_csv_boto(
                            self.spark, self.args, pd.DataFrame({"a": [1]}), self.logger
                        )
                        client.put_object.assert_called_once()
                    else:
                        with self.assertRaises(ClientError):
                            driver.append_csv_boto(
                                self.spark, self.args, pd.DataFrame({"a": [1]}), self.logger
                            )
                        client.put_object.assert_not_called()
                client.close.assert_called_once()

    def test_metadata_count_and_datetime_fields(self):
        frame = Mock()
        frame.count.return_value = 42
        driver.write_metadata(self.spark, frame, self.args, "s3a://data/meta", self.logger)
        metadata = self.spark.createDataFrame.call_args.args[0]
        self.assertEqual(metadata.iloc[0]["inc_rowcount"], 42)
        self.assertEqual(metadata.iloc[0]["start_time"], "2026-09-12")
        self.assertRegex(metadata.iloc[0]["finish_time"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertEqual(self.args["meta_variables"]["inc_rowcount"], "None")

    def test_skip_count_when_metric_has_explicit_value(self):
        frame = Mock()
        self.args["meta_variables"] = {"inc_rowcount": 12}
        driver.write_metadata(self.spark, frame, self.args, "s3a://data/meta", self.logger)
        frame.count.assert_not_called()

    def test_boto_metadata_does_not_create_spark_dataframe(self):
        self.args.update(use_parquet_for_meta=False, no_delete_meta_file=True)
        with patch.object(driver, "append_csv_boto") as append:
            driver.write_metadata(
                self.spark,
                Mock(count=Mock(return_value=3)),
                self.args,
                "s3a://data/meta",
                self.logger,
            )
        append.assert_called_once()
        self.spark.createDataFrame.assert_not_called()

    def test_connection_timeout_is_converted_to_milliseconds(self):
        module = types.ModuleType("pyspark")
        module.SparkConf = Mock()
        self.args.update(
            attempts_maximum=1, retry_interval=60, connection_timeout=300, connection_ttl=300
        )
        with patch.dict("sys.modules", {"pyspark": module}):
            conf = driver.build_spark_conf(self.args)
        conf.set.assert_any_call("spark.hadoop.fs.s3a.connection.timeout", "300000")

    def test_sharded_write_preserves_partition_by(self):
        frame = Mock()
        self.args.update(clear_s3_path=False, partition_by=["shard_id"], repartition=4)
        driver.write_data(self.spark, frame, self.args, "s3a://data/dataset")
        frame.repartition.assert_called_once_with(4)
        writer = frame.repartition.return_value.write.mode.return_value
        writer.partitionBy.assert_called_once_with("shard_id")
        writer.partitionBy.return_value.parquet.assert_called_once_with("s3a://data/dataset")

    def test_spark_csv_writes_before_deleting_original(self):
        events = []
        fs = Mock()

        def path(value):
            item = Mock()
            item.toString.return_value = value
            item.getFileSystem.return_value = fs
            return item

        self.spark._jvm.org.apache.hadoop.fs.Path.side_effect = path
        fs.exists.return_value = True
        part = Mock()
        part.getPath.return_value.getName.return_value = "part-0.csv"
        fs.listStatus.return_value = [part]
        result = self.spark.read.schema.return_value.csv.return_value.unionByName.return_value
        result.coalesce.return_value.write.mode.return_value.option.return_value.csv.side_effect = (
            lambda value: events.append("write")
        )
        fs.delete.side_effect = lambda *args: events.append("delete") or True
        fs.rename.side_effect = lambda *args: events.append("rename") or True
        driver.append_csv_spark(
            self.spark, self.args, Mock(), "s3a://data/meta", False, self.logger
        )
        self.assertEqual(events, ["write", "delete", "rename", "delete"])

    def test_spark_csv_write_failure_keeps_original(self):
        fs = Mock()
        self.spark._jvm.org.apache.hadoop.fs.Path.return_value.getFileSystem.return_value = fs
        fs.exists.return_value = True
        result = self.spark.read.schema.return_value.csv.return_value.unionByName.return_value
        result.coalesce.return_value.write.mode.return_value.option.return_value.csv.side_effect = (
            RuntimeError("write failed")
        )
        with self.assertRaisesRegex(RuntimeError, "write failed"):
            driver.append_csv_spark(
                self.spark, self.args, Mock(), "s3a://data/meta", False, self.logger
            )
        fs.delete.assert_not_called()
        fs.rename.assert_not_called()
