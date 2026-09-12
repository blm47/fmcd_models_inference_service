"""Запуск инференса и ожидание всех шардов в Airflow 2.6.3."""

from __future__ import annotations

import json
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from airflow.exceptions import AirflowException, AirflowRescheduleException, AirflowTaskTimeout
from airflow.sensors.base import BaseSensorOperator, PokeReturnValue


class GPUModelInferenceOperator(BaseSensorOperator):
    """
    Отправляет запросы в сервис и освобождает worker между проверками.

    Без partition_by и n_shards создаёт один расчёт. С ними использует пути
    prefix/partition_by=0 ... prefix/partition_by=n_shards-1 на входе и выходе.
    Успех — только DONE у всех шардов. Ошибка запуска, FAILED или ABORTED
    завершает задачу Airflow с ошибкой и запрашивает отмену остальных расчётов.

    Автоматические retries запрещены: перед ручным Clear Поддержка проверяет
    остановку старых расчётов и очищает выходной префикс отдельной задачей.
    Ключи запросов стабильны между reschedule и меняются с номером попытки TI.
    Состояние не зависит от XCom, очищаемого Airflow при reschedule.
    """

    template_fields = (
        "service_url",
        "s3_input_prefix",
        "s3_output_prefix",
        "model_name",
        "partition_by",
        "n_shards",
    )
    ui_color = "#d8ebfa"

    def __init__(
        self,
        *,
        service_url: str,
        s3_input_prefix: str,
        s3_output_prefix: str,
        model_name: str,
        partition_by: str | None = None,
        n_shards: int | None = None,
        poke_interval: float = 60,
        timeout: float = 86400,
        request_timeout: float = 30,
        **kwargs,
    ):
        # Не наследуем автоматические retries из default_args DAG.
        kwargs.setdefault("retries", 0)
        kwargs.setdefault("mode", "reschedule")
        if kwargs["retries"] != 0 or kwargs["mode"] != "reschedule":
            raise ValueError("Нужны retries=0 и mode='reschedule': повтор выполняется вручную")
        if kwargs.get("soft_fail") or kwargs.get("silent_fail"):
            raise ValueError(
                "soft_fail и silent_fail скрывают ошибки инференса и не поддерживаются"
            )
        if request_timeout <= 0:
            raise ValueError("request_timeout должен быть положительным")
        super().__init__(poke_interval=poke_interval, timeout=timeout, **kwargs)
        self.service_url = service_url
        self.s3_input_prefix = s3_input_prefix
        self.s3_output_prefix = s3_output_prefix
        self.model_name = model_name
        self.partition_by = partition_by
        self.n_shards = n_shards
        self.request_timeout = request_timeout
        self._tasks = {}

    def _requests(self, context):
        """Строит ключи по TI и попытке после обработки Jinja-шаблонов."""
        if urlsplit(self.service_url).scheme not in {"http", "https"}:
            raise ValueError("service_url должен начинаться с http:// или https://")
        if not self.model_name:
            raise ValueError("model_name не должен быть пустым")
        for prefix in (self.s3_input_prefix, self.s3_output_prefix):
            parsed = urlsplit(prefix)
            if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
                raise ValueError("Нужен непустой S3-префикс вида s3://bucket/path")
        if (self.partition_by is None) != (self.n_shards is None):
            raise ValueError("partition_by и n_shards нужно передать вместе")
        suffixes = [""]
        if self.partition_by is not None:
            if not self.partition_by or any(c in self.partition_by for c in "/=?#"):
                raise ValueError("partition_by должен содержать имя колонки")
            count = int(self.n_shards)
            if isinstance(self.n_shards, bool) or str(count) != str(self.n_shards) or count < 1:
                raise ValueError("n_shards должен быть положительным целым числом")
            suffixes = [f"/{self.partition_by}={i}" for i in range(count)]
        ti = context["ti"]
        requests = []
        for suffix in suffixes:
            identity = json.dumps(
                [ti.dag_id, ti.run_id, ti.task_id, ti.map_index, ti.try_number, suffix]
            )
            requests.append(
                {
                    "idempotency_key": str(uuid.uuid5(uuid.NAMESPACE_URL, identity)),
                    "model_name": self.model_name,
                    "s3_input_path": self.s3_input_prefix.rstrip("/") + suffix,
                    "s3_output_path": self.s3_output_prefix.rstrip("/") + suffix,
                }
            )
        return requests

    def _http(self, method, path, payload=None):
        """Повторяет временные HTTP-ошибки с тем же телом и ключом запроса."""
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        for attempt in range(3):
            try:
                request = Request(
                    self.service_url.rstrip("/") + path,
                    data=data,
                    method=method,
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(request, timeout=self.request_timeout) as response:
                    result = json.load(response)
                if not isinstance(result, dict):
                    raise AirflowException(f"Некорректный ответ {method} {path}")
                return result
            except HTTPError as exc:
                error = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:1000]}"
                if exc.code not in {429, 500, 502, 503, 504}:
                    raise AirflowException(f"{method} {path}: {error}") from exc
            except (URLError, TimeoutError, ConnectionError) as exc:
                error = str(exc)
            if attempt < 2:
                self.log.warn(f"Повтор HTTP-запроса {method} {path}: {error}")
                time.sleep(1)
        raise AirflowException(f"{method} {path}: {error}")

    def _abort_remaining(self):
        """Отправляет отмену всем известным незавершённым задачам, даже при ошибках API."""
        for task_id, task in self._tasks.items():
            if task.get("status") in {"DONE", "FAILED", "ABORTED"}:
                continue
            try:
                result = self._http("POST", f"/tasks/{quote(task_id, safe='')}/abort")
                self.log.warn(f"Отмена {task_id}: статус={result.get('status')}")
            except Exception as exc:
                self.log.error(f"Не удалось отменить {task_id}, требуется ручная проверка: {exc}")

    def execute(self, context):
        self._tasks = {}
        try:
            return super().execute(context)
        except AirflowRescheduleException:
            # Штатное освобождение worker не является ошибкой и не отменяет расчёты.
            raise
        except (Exception, AirflowTaskTimeout):
            self._abort_remaining()
            raise

    def poke(self, context):
        """Восстанавливает заявки по ключам и проверяет завершение всех шардов."""
        errors = []
        for payload in self._requests(context):
            try:
                # Повторный POST возвращает прежний task_id, включая завершённую задачу.
                task = self._http("POST", "/infer", payload)
                task_id = task["task_id"]
                if not isinstance(task_id, str) or not task_id:
                    raise ValueError("Сервис вернул пустой task_id")
                self._tasks[task_id] = task
                task = self._http("GET", f"/tasks/{quote(task_id, safe='')}/status")
                self._tasks[task_id] = task
                status = task["status"]
                self.log.info(f"Инференс {task_id}: {status}, выход={payload['s3_output_path']}")
                if status in {"FAILED", "ABORTED", "ABORTING"}:
                    errors.append(f"{task_id}: {status}, причина={task.get('error')}")
                elif status not in {"QUEUED", "RUNNING", "FINALIZING", "DONE"}:
                    errors.append(f"{task_id}: неизвестный статус {status}")
            except Exception as exc:
                errors.append(f"Ключ {payload['idempotency_key']}: {exc}")
                self.log.error(
                    f"Не удалось проверить шард {payload['s3_output_path']}: {exc}. "
                    f"Ключ запроса: {payload['idempotency_key']}"
                )
            # Проходим все шарды, чтобы восстановить ID предыдущего poke для отмены.
        if errors:
            raise AirflowException("Ошибка инференса: " + "; ".join(errors))
        if all(task["status"] == "DONE" for task in self._tasks.values()):
            return PokeReturnValue(
                True,
                {
                    "task_ids": list(self._tasks),
                    "s3_output_prefix": self.s3_output_prefix.rstrip("/"),
                },
            )
        return False

    def on_kill(self):
        """Запрашивает отмену при штатном завершении активного процесса Airflow."""
        self._abort_remaining()
