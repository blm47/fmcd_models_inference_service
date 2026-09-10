import unittest
from unittest.mock import Mock, patch

from app.tasks.maintenance import TaskMaintenance


class MaintenanceTests(unittest.TestCase):
    def test_cleanup_runs_without_requests_and_retries_after_failure(self):
        store = Mock()
        store.cleanup.side_effect = [OSError("S3 unavailable"), None]
        maintenance = TaskMaintenance(store, 3600)
        maintenance._stop = Mock()
        maintenance._stop.wait.side_effect = [False, False, True]
        with patch("app.tasks.maintenance.random.uniform", return_value=1.0):
            with self.assertLogs("app.tasks.maintenance", level="ERROR"):
                maintenance._run()
        self.assertEqual(store.cleanup.call_count, 2)
        self.assertEqual([call.args[0] for call in maintenance._stop.wait.call_args_list], [1.0, 3600, 3600])

    def test_stop_interrupts_wait_and_joins_thread(self):
        maintenance = TaskMaintenance(Mock(), 3600)
        with patch("app.tasks.maintenance.random.uniform", return_value=3600):
            maintenance.start()
            maintenance.stop()
        self.assertFalse(maintenance._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
