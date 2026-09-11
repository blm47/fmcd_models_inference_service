"""Проверки совместимости с logger, у которого нет API стандартного logging."""

import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

from logger_stub import make_logger

from app.core.logging import setup_logging
from app.tasks.worker import _keep_heartbeat


class LoggingTests(unittest.TestCase):
    def test_setup_returns_library_singleton_without_configuring_it(self):
        logger = make_logger()
        package = types.ModuleType("dadm_functions")
        module = types.ModuleType("dadm_functions.logger")
        module.logger = logger
        with patch.dict(sys.modules, {"dadm_functions": package, "dadm_functions.logger": module}):
            self.assertIs(setup_logging(), logger)
            self.assertIs(setup_logging(), logger)
        self.assertEqual(logger.mock_calls, [])

    def test_error_receives_one_string_with_traceback(self):
        logger = make_logger()
        stop = Mock(spec=threading.Event)
        stop.wait.side_effect = [False, True]
        store = Mock()
        store.heartbeat.side_effect = RuntimeError("Ошибка подключения S3")
        _keep_heartbeat("task-1", store, stop, logger)
        logger.error.assert_called_once()
        args, kwargs = logger.error.call_args
        self.assertEqual(len(args), 1)
        self.assertEqual(kwargs, {})
        self.assertIn("task-1", args[0])
        self.assertIn("Traceback", args[0])
        self.assertIn("RuntimeError: Ошибка подключения S3", args[0])
