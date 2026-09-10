import io
import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from unittest.mock import Mock, patch

from app.tasks.backends.s3_backend import (
    LockAcquireTimeoutError,
    LockLostError,
    S3LeaseLockBackend,
)
from app.tasks.state import TaskState, TaskStatus


STATE_KEY = "bucket/task_state.json"
LOCK_KEY = "bucket/task_state.lock"
MONTH = 30 * 24 * 60 * 60


class MemoryS3:
    """Хранит сериализованные байты и фиксирует запись при close, как файл S3."""

    def __init__(self):
        self.objects = {}
        self.reads = []
        self.writes = []
        self.before_write = None

    def open(self, key, mode):
        if mode == "rb":
            self.reads.append(key)
            if key not in self.objects:
                raise FileNotFoundError(key)
            return io.BytesIO(self.objects[key])
        if mode != "wb":
            raise AssertionError(f"Unexpected S3 operation: {mode}")
        fs = self

        class Writer(io.BytesIO):
            def close(self):
                try:
                    if not self.closed:
                        body = self.getvalue()
                        if fs.before_write is not None:
                            fs.before_write(key, body)
                        fs.objects[key] = body
                        fs.writes.append((key, body))
                finally:
                    super().close()

        return Writer()

    def put(self, key, payload):
        self.objects[key] = json.dumps(payload).encode()

    def get(self, key):
        return json.loads(self.objects[key])


class FakeClock:
    def __init__(self):
        self.wall = 10 * MONTH
        self.monotonic = 0.0
        self.sleeps = []
        self.on_sleep = None

    def advance(self, seconds):
        self.wall += seconds
        self.monotonic += seconds

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.advance(seconds)
        if self.on_sleep is not None:
            self.on_sleep(seconds)


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.fs = MemoryS3()
        self.backend = S3LeaseLockBackend(
            self.fs, STATE_KEY, LOCK_KEY, wait_timeout_sec=3, retention_months=1
        )
        self.clock = FakeClock()
        patches = self.enterContext(ExitStack())
        patches.enter_context(patch("app.tasks.backends.s3_backend.time.time", side_effect=lambda: self.clock.wall))
        patches.enter_context(patch("app.tasks.backends.s3_backend.time.monotonic", side_effect=lambda: self.clock.monotonic))
        patches.enter_context(patch("app.tasks.backends.s3_backend.time.sleep", side_effect=self.clock.sleep))
        self.jitter = patches.enter_context(patch("app.tasks.backends.s3_backend.random.uniform", return_value=1.0))

    def competitor(self, **changes):
        lock = {
            "owner_id": "another-pod",
            "token": "another-attempt",
            "acquired_at": self.clock.wall,
            "expires_at": self.clock.wall + 60,
        }
        lock.update(changes)
        self.fs.put(LOCK_KEY, lock)
        return lock

    def task(self, task_id="task", **changes):
        fields = dict(
            task_id=task_id,
            model_name="model",
            s3_input_path="input",
            s3_output_path="output",
            status=TaskStatus.RUNNING,
            total_rows=10,
            started_at=self.clock.wall,
            updated_at=self.clock.wall,
        )
        fields.update(changes)
        return TaskState(**fields)

    def seed(self, *tasks):
        self.fs.put(STATE_KEY, self.backend._serialize_state({task.task_id: task for task in tasks}))

    def assert_no_state_write(self):
        self.assertFalse(any(key == STATE_KEY for key, _ in self.fs.writes))

    def test_waits_two_seconds_before_reading_or_mutating_state(self):
        change = Mock(return_value={})

        def while_waiting(seconds):
            self.assertEqual(seconds, 2.0)
            change.assert_not_called()
            self.assertNotIn(STATE_KEY, self.fs.reads)
            self.assert_no_state_write()

        self.clock.on_sleep = while_waiting
        self.assertEqual(self.backend.mutate(change), {})
        change.assert_called_once_with({})
        self.assertEqual(self.fs.get(STATE_KEY), {"tasks": {}})
        self.assertEqual(self.clock.sleeps, [2.0])
        self.assertEqual(self.fs.get(LOCK_KEY)["expires_at"], 0.0)
        self.assertEqual(set(self.fs.objects), {STATE_KEY, LOCK_KEY})

    def test_lock_stolen_during_confirmation_is_not_used_or_released(self):
        change = Mock()
        foreign = {}

        def steal(seconds):
            if not foreign:
                foreign.update(self.competitor())

        self.clock.on_sleep = steal
        with self.assertRaises(LockAcquireTimeoutError):
            self.backend.mutate(change)
        change.assert_not_called()
        self.assert_no_state_write()
        self.assertEqual(self.fs.get(LOCK_KEY), foreign)

    def test_expired_or_missing_confirmation_never_runs_callback(self):
        for failure in ("expired", "missing"):
            with self.subTest(failure=failure):
                self.fs.objects.clear()
                self.fs.writes.clear()
                change = Mock()

                def interfere(seconds):
                    if LOCK_KEY in self.fs.objects:
                        if failure == "expired":
                            candidate = self.fs.get(LOCK_KEY)
                            candidate["expires_at"] = self.clock.wall
                            self.fs.put(LOCK_KEY, candidate)
                        else:
                            del self.fs.objects[LOCK_KEY]

                self.clock.on_sleep = interfere
                with self.assertRaises(LockAcquireTimeoutError):
                    self.backend.mutate(change)
                change.assert_not_called()
                self.assert_no_state_write()

    def test_attempt_is_retried_with_new_token_after_failed_confirmation(self):
        self.backend._wait_timeout_sec = 10
        attempts = []

        def interfere(seconds):
            if seconds == 2:
                attempts.append(self.fs.get(LOCK_KEY)["token"])
                if len(attempts) == 1:
                    self.competitor()
            elif len(attempts) == 1:
                self.competitor(expires_at=0)

        self.clock.on_sleep = interfere
        self.backend.mutate(lambda current: current)
        self.assertEqual(len(attempts), 2)
        self.assertNotEqual(*attempts)

    def test_every_successful_acquisition_uses_a_new_token(self):
        self.backend.mutate(lambda current: current)
        first = self.fs.get(LOCK_KEY)["token"]
        self.backend.mutate(lambda current: current)
        self.assertNotEqual(first, self.fs.get(LOCK_KEY)["token"])

    def test_ownership_rechecked_after_state_read_before_callback(self):
        change = Mock()
        read_state = self.backend.read_state

        def delayed_read():
            state = read_state()
            self.competitor()
            return state

        with patch.object(self.backend, "read_state", side_effect=delayed_read):
            with self.assertRaises(LockLostError):
                self.backend.mutate(change)
        change.assert_not_called()
        self.assert_no_state_write()
        self.assertEqual(self.fs.get(LOCK_KEY)["owner_id"], "another-pod")

    def test_same_owner_different_token_is_also_lock_loss(self):
        def change(state):
            self.competitor(owner_id=self.backend._owner_id)
            return state

        with self.assertRaises(LockLostError):
            self.backend.mutate(change)
        self.assert_no_state_write()
        self.assertEqual(self.fs.get(LOCK_KEY)["token"], "another-attempt")

    def test_lost_lock_after_callback_does_not_replay_callback(self):
        def change(state):
            self.competitor()
            return state

        callback = Mock(side_effect=change)
        with self.assertRaises(LockLostError):
            self.backend.mutate(callback)
        callback.assert_called_once()
        self.assert_no_state_write()

    def test_expiry_during_callback_prevents_write_and_release(self):
        def change(state):
            self.clock.advance(60)
            return state

        with self.assertRaises(LockLostError):
            self.backend.mutate(change)
        self.assert_no_state_write()
        self.assertEqual(len(self.fs.writes), 1)
        self.assertLess(self.fs.get(LOCK_KEY)["expires_at"], self.clock.wall)

    def test_small_remaining_lease_refuses_write(self):
        def change(state):
            self.clock.advance(54)
            return state

        with self.assertRaises(LockLostError):
            self.backend.mutate(change)
        self.assert_no_state_write()

    def test_state_serialization_precedes_last_ownership_check(self):
        serialize = self.backend._serialize_state

        def slow_serialize(state):
            body = serialize(state)
            self.competitor()
            return body

        with patch.object(self.backend, "_serialize_state", side_effect=slow_serialize):
            with self.assertRaises(LockLostError):
                self.backend.mutate(lambda current: current)
        self.assert_no_state_write()

    def test_json_encoding_precedes_last_ownership_check(self):
        encode = self.backend._encode_json

        def slow_encode(payload):
            body = encode(payload)
            if "tasks" in payload:
                self.clock.advance(60)
            return body

        with patch.object(self.backend, "_encode_json", side_effect=slow_encode):
            with self.assertRaises(LockLostError):
                self.backend.mutate(lambda current: current)
        self.assert_no_state_write()

    def test_old_acquired_at_does_not_override_unexpired_lease(self):
        foreign = self.competitor(acquired_at=self.clock.wall - 100000)
        with self.assertRaises(LockAcquireTimeoutError):
            self.backend.mutate(lambda current: current)
        self.assertEqual(self.fs.writes, [])
        self.assertEqual(self.fs.get(LOCK_KEY), foreign)

    def test_timeout_uses_monotonic_clock_when_wall_clock_moves_back(self):
        self.competitor()

        def jump_back(seconds):
            self.clock.wall -= 3600

        self.clock.on_sleep = jump_back
        with self.assertRaises(LockAcquireTimeoutError):
            self.backend.mutate(lambda current: current)
        self.assertEqual(self.clock.monotonic, 3.0)
        self.assertEqual(self.fs.writes, [])

    def test_random_polling_is_bounded_by_remaining_deadline(self):
        self.competitor()
        self.jitter.return_value = 1.25
        with self.assertRaises(LockAcquireTimeoutError):
            self.backend.mutate(lambda current: current)
        self.jitter.assert_called_with(0.5, 1.5)
        self.assertEqual(self.clock.sleeps, [1.25, 1.25, 0.5])

    def test_release_error_does_not_hide_success_or_original_failure(self):
        def fail_release(key, body):
            if key == LOCK_KEY and json.loads(body)["expires_at"] == 0:
                raise OSError("release failed")

        self.fs.before_write = fail_release
        with self.assertLogs("app.tasks.backends.s3_backend", level="WARNING"):
            self.assertEqual(self.backend.mutate(lambda current: current), {})
        self.assertIn(STATE_KEY, self.fs.objects)
        self.clock.advance(61)

        def fail_mutation(state):
            raise ValueError("original failure")

        with self.assertLogs("app.tasks.backends.s3_backend", level="WARNING"):
            with self.assertRaisesRegex(ValueError, "original failure"):
                self.backend.mutate(fail_mutation)

    def test_read_state_does_not_read_or_write_the_lock(self):
        self.seed(self.task())
        self.assertEqual(set(self.backend.read_state()), {"task"})
        self.assertEqual(self.fs.reads, [STATE_KEY])
        self.assertEqual(self.fs.writes, [])

    def test_initialize_creates_exactly_two_objects_once(self):
        self.backend.initialize()
        self.assertEqual(set(self.fs.objects), {STATE_KEY, LOCK_KEY})
        self.fs.writes.clear()
        self.backend.initialize()
        self.assertEqual(self.fs.writes, [])

    def test_initialize_missing_lock_preserves_existing_tasks(self):
        self.seed(self.task())
        self.backend.initialize()
        self.assertEqual(set(self.fs.objects), {STATE_KEY, LOCK_KEY})
        self.assertEqual(set(self.backend.read_state()), {"task"})

    def test_noop_cleanup_is_read_only_even_with_no_state_file(self):
        self.assertFalse(self.backend.cleanup())
        self.seed(self.task())
        self.assertFalse(self.backend.cleanup())
        self.assertEqual(self.fs.reads, [STATE_KEY, STATE_KEY])
        self.assertEqual(self.fs.writes, [])

    def test_retention_uses_completion_or_owner_heartbeat_instead_of_start(self):
        old = self.clock.wall - 2 * MONTH
        recent = self.clock.wall
        tasks = [
            self.task("running", started_at=old, heartbeat_at=recent),
            self.task("old-running", heartbeat_at=old, updated_at=recent),
            self.task("legacy-running", started_at=old, updated_at=recent),
            self.task("old-legacy", updated_at=old),
            self.task("done", status=TaskStatus.DONE, started_at=old, finished_at=recent),
            self.task("old-done", status=TaskStatus.DONE, finished_at=old, updated_at=recent),
            self.task("failed", status=TaskStatus.FAILED, updated_at=old),
            self.task("aborted", status=TaskStatus.ABORTED, finished_at=recent),
        ]
        self.seed(*tasks)
        self.assertTrue(self.backend.cleanup())
        self.assertEqual(set(self.backend.read_state()), {"running", "legacy-running", "done", "aborted"})
        self.assertEqual(set(self.fs.objects), {STATE_KEY, LOCK_KEY})

    def test_cleanup_rechecks_refreshed_tasks_after_preflight(self):
        self.seed(self.task(updated_at=self.clock.wall - 2 * MONTH))
        acquire = self.backend._try_acquire

        def refresh(deadline=None):
            self.seed(self.task())
            return acquire(deadline)

        with patch.object(self.backend, "_try_acquire", side_effect=refresh):
            self.assertTrue(self.backend.cleanup())
        self.assertIn("task", self.backend.read_state())

    def test_legacy_task_state_deserializes_without_new_fields(self):
        legacy = self.backend._serialize_state({"task": self.task()})
        for field in ("heartbeat_at", "owner_id", "cancel_requested"):
            legacy["tasks"]["task"].pop(field, None)
        self.fs.put(STATE_KEY, legacy)
        task = self.backend.read_state()["task"]
        self.assertIsNone(task.heartbeat_at)
        self.assertFalse(task.cancel_requested)


class LocalSerializationTests(unittest.TestCase):
    def test_threads_cannot_overlap_read_modify_write(self):
        fs = MemoryS3()
        backend = S3LeaseLockBackend(fs, STATE_KEY, LOCK_KEY)
        running = threading.Event()
        finish = threading.Event()
        second_started = threading.Event()

        def first(state):
            running.set()
            self.assertTrue(finish.wait(2))
            return state

        def second(state):
            second_started.set()
            return state

        with patch("app.tasks.backends.s3_backend.LOCK_CONFIRMATION_DELAY_SECONDS", 0.01):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first_future = pool.submit(backend.mutate, first)
                self.assertTrue(running.wait(2))
                second_future = pool.submit(backend.mutate, second)
                self.assertFalse(second_started.wait(0.05))
                finish.set()
                first_future.result(timeout=2)
                second_future.result(timeout=2)
        self.assertTrue(second_started.is_set())
        state_writes = [key for key, _ in fs.writes if key == STATE_KEY]
        self.assertEqual(len(state_writes), 2)

    def test_timeout_includes_wait_for_local_lock(self):
        fs = MemoryS3()
        backend = S3LeaseLockBackend(fs, STATE_KEY, LOCK_KEY, wait_timeout_sec=0.01)
        backend._local_lock.acquire()
        try:
            with self.assertRaises(LockAcquireTimeoutError):
                backend.mutate(lambda current: current)
        finally:
            backend._local_lock.release()
        self.assertEqual(fs.reads, [])
        self.assertEqual(fs.writes, [])


class ConfigurationTests(unittest.TestCase):
    def test_rejects_nonpositive_nonfinite_and_too_short_lease(self):
        for setting, value in (
            ("lease_seconds", 7),
            ("lease_seconds", 0),
            ("lease_seconds", float("inf")),
            ("wait_timeout_sec", 0),
            ("wait_timeout_sec", float("nan")),
            ("poll_interval_sec", -1),
            ("retention_months", 0),
        ):
            with self.subTest(setting=setting, value=value):
                with self.assertRaises(ValueError):
                    S3LeaseLockBackend(None, STATE_KEY, LOCK_KEY, **{setting: value})

    def test_rejects_overlapping_or_empty_object_keys(self):
        for state_key, lock_key in (("", LOCK_KEY), (STATE_KEY, ""), (STATE_KEY, STATE_KEY)):
            with self.subTest(state_key=state_key, lock_key=lock_key):
                with self.assertRaises(ValueError):
                    S3LeaseLockBackend(None, state_key, lock_key)


if __name__ == "__main__":
    unittest.main()
