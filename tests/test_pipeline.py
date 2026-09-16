"""
Проверки общего batching и последовательного lifecycle без GPU.
"""

import gc
import threading
import types
import unittest
import weakref
from dataclasses import replace
from unittest.mock import Mock, patch

import pandas as pd
from fake_s3 import FakeS3, make_store
from logger_stub import make_logger

from app.models.contracts import ModelBundle, ModelSpec
from app.models.inference import run_inference_on_chunk
from app.models.registry import BACKENDS, create_bundle
from app.tasks.state import TaskStatus
from app.tasks.worker import consume_queue


def make_spec():
    return ModelSpec("cc", "fmcd_cc_dc", "missing/cc", ("id",), "cpu", 2, 3)


class TestBundle(ModelBundle):
    def load(self):
        self.required_columns = ("value",)
        self.output_columns = ("score",)

    def predict_batch(self, frame, check_shutdown):
        return pd.DataFrame({"id": frame.id.to_numpy(), "score": frame.value.to_numpy() * 2})

    def close(self):
        pass


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.bundle = TestBundle(make_spec(), make_logger())
        self.bundle.load()
        self.frame = pd.DataFrame({"id": [9, 9, 4], "value": [1.0, 2.0, 3.0]}, index=[8, 3, 3])

    def test_tail_single_row_and_chunk_smaller_than_batch(self):
        for size in (1, 2, 5):
            with self.subTest(size=size):
                checkpoint = Mock()
                with patch.object(
                    self.bundle, "predict_batch", wraps=self.bundle.predict_batch
                ) as p:
                    result = run_inference_on_chunk(self.frame, self.bundle, size, checkpoint)
                pd.testing.assert_frame_equal(
                    result, pd.DataFrame({"id": [9, 9, 4], "score": [2.0, 4.0, 6.0]})
                )
                self.assertTrue(all(len(call.args[0]) <= size for call in p.call_args_list))
                self.assertEqual(checkpoint.call_count, 2 * p.call_count)

    def test_invalid_result_is_rejected(self):
        for kind in ("rows", "keys", "columns", "input", "types"):
            calls = 0

            def predict(frame, check_shutdown, kind=kind):
                nonlocal calls
                calls += 1
                result = TestBundle.predict_batch(self.bundle, frame, check_shutdown)
                if kind == "rows":
                    return result.iloc[:0]
                if kind == "keys":
                    result["id"] += 1
                if kind == "columns":
                    result["extra"] = 1
                if kind == "input":
                    frame["value"] = 0
                if kind == "types" and calls > 1:
                    result["score"] = result.score.astype(str)
                return result

            with self.subTest(kind=kind), patch.object(self.bundle, "predict_batch", predict):
                with self.assertRaises(ValueError):
                    run_inference_on_chunk(self.frame, self.bundle, 2, lambda: None)

    def test_backend_reused_without_instance_cache(self):
        module = types.SimpleNamespace(TestBundle=TestBundle)
        with (
            patch.dict(BACKENDS, {"test": "test_backend:TestBundle"}),
            patch("app.models.registry.import_module", return_value=module),
        ):
            first = create_bundle(replace(make_spec(), backend="test"), make_logger())
            second = create_bundle(
                replace(make_spec(), backend="test", name="dc", artifacts_dir="missing/dc"),
                make_logger(),
            )
        self.assertIsNot(first, second)
        self.assertEqual(first.required_columns, ())
        self.assertNotEqual(first.spec.artifacts_dir, second.spec.artifacts_dir)

    def test_invest_stub_fails_explicitly_and_can_be_closed(self):
        bundle = create_bundle(
            replace(make_spec(), name="fmcd_invest", backend="fmcd_invest"), make_logger()
        )
        with self.assertRaisesRegex(NotImplementedError, "pipeline ещё не реализован"):
            bundle.load()
        bundle.close()
        bundle.close()

    def test_consumer_waits_for_close_and_does_not_keep_bundle_alive(self):
        store = make_store(FakeS3())
        store.initialize()
        tasks = [
            store.enqueue(str(i), "cc", "s3://in/data", f"s3://out/{i}", None) for i in range(2)
        ]
        stop = threading.Event()
        events = []
        refs = []
        client = Mock()
        client.get_schema_columns.return_value = {"id", "value"}
        client.count_rows.return_value = 3
        client.prefix_has_results.return_value = False
        client.iter_chunks.side_effect = lambda *args: iter([self.frame])

        class LifecycleBundle(TestBundle):
            def load(self):
                events.append("load")
                super().load()

            def close(self):
                events.append("close")
                if events.count("close") == 2:
                    stop.set()

        def factory(spec, logger):
            gc.collect()
            self.assertTrue(all(ref() is None for ref in refs))
            events.append("create")
            bundle = LifecycleBundle(spec, logger)
            refs.append(weakref.ref(bundle))
            return bundle

        timer = threading.Timer(5, stop.set)
        timer.start()
        self.addCleanup(timer.cancel)
        with patch("app.tasks.worker.create_bundle", side_effect=factory):
            consume_queue(
                store,
                {"cc": make_spec()},
                types.SimpleNamespace(task_store=store.config),
                client,
                "pod",
                stop,
                make_logger(),
            )
        self.assertEqual(events, ["create", "load", "close"] * 2)
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertTrue(all(store.get(task.task_id).status == TaskStatus.DONE for task in tasks))
        self.assertEqual(client.get_writer.return_value.write_chunk.call_count, 2)
