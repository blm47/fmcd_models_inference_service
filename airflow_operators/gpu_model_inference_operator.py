"""
Запуск инференса и ожидание всех шардов в Airflow 2.6.3.
"""

from __future__ import annotations

import json
import ssl
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

    Без shard_id_column и num_shards создаёт один расчёт. С ними использует пути
    prefix/shard_id_column=0 ... prefix/shard_id_column=num_shards-1 на входе и выходе.
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
        "shard_id_column",
        "num_shards",
        "calc_utilization",
    )
    ui_color = "#d8ebfa"

    def __init__(
        self,
        *,
        service_url: str,
        s3_input_prefix: str,
        s3_output_prefix: str,
        model_name: str,
        shard_id_column: str | None = None,
        num_shards: int | None = None,
        poke_interval: float = 60,
        timeout: float = 86400,
        request_timeout: float = 30,
        verify_ssl: bool = False,
        calc_utilization: bool | str = False,
        **kwargs,
    ):
        """
        :param service_url: Базовый URL сервиса инференса с http:// или https://.
        :param s3_input_prefix: Префикс входных данных вида s3://bucket/path.
        :param s3_output_prefix: Префикс результатов вида s3://bucket/path.
        :param model_name: Непустое имя модели для запуска инференса.
        :param shard_id_column: Имя колонки идентификатора шарда; задаётся вместе с num_shards.
            По умолчанию None: запускается один расчёт без разбиения на шарды.
        :param num_shards: Положительное целое число шардов; задаётся вместе с shard_id_column.
            К входному и выходному префиксам добавляется /shard_id_column=i,
            где i от 0 до num_shards - 1. По умолчанию None.
        :param poke_interval: Интервал между проверками статуса в секундах,
            по умолчанию 60.
        :param timeout: Общий таймаут ожидания завершения в секундах, включая время
            между reschedule, по умолчанию 86400.
        :param request_timeout: Положительный таймаут отдельного HTTP-запроса
            в секундах, по умолчанию 30.
        :param verify_ssl: Проверять сертификат и имя HTTPS-сервиса, по умолчанию False.
            False отключает проверку для запуска, опроса и отмены инференса.
        :param calc_utilization: Считать утилизацию в памяти worker и вывести итог
            в logger, по умолчанию False. Допускается Jinja-шаблон; после рендеринга
            ожидается bool или строка true/false/1/0.
        :param kwargs: Дополнительные параметры BaseSensorOperator, включая task_id.
            retries должен быть 0, mode — 'reschedule'; эти значения заданы
            по умолчанию. soft_fail и silent_fail не должны быть включены.
        """
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
        if not isinstance(calc_utilization, (bool, str)):
            raise ValueError("calc_utilization должен быть bool или Jinja-шаблоном")
        if not isinstance(verify_ssl, bool):
            raise ValueError("verify_ssl должен быть bool")
        super().__init__(poke_interval=poke_interval, timeout=timeout, **kwargs)
        self.service_url = service_url
        self.s3_input_prefix = s3_input_prefix
        self.s3_output_prefix = s3_output_prefix
        self.model_name = model_name
        self.shard_id_column = shard_id_column
        self.num_shards = num_shards
        self.request_timeout = request_timeout
        self.verify_ssl = verify_ssl
        self.calc_utilization = calc_utilization
        self._tasks = {}

    def _requests(self, context):
        """
        Строит ключи по TI и попытке после обработки Jinja-шаблонов.
        """
        if isinstance(self.calc_utilization, str):
            value = self.calc_utilization.strip().lower()
            if value not in {"true", "false", "1", "0"}:
                raise ValueError("calc_utilization после Jinja должен быть bool")
            self.calc_utilization = value in {"true", "1"}
        if urlsplit(self.service_url).scheme not in {"http", "https"}:
            raise ValueError("service_url должен начинаться с http:// или https://")
        if not self.model_name:
            raise ValueError("model_name не должен быть пустым")
        for prefix in (self.s3_input_prefix, self.s3_output_prefix):
            parsed = urlsplit(prefix)
            if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
                raise ValueError("Нужен непустой S3-префикс вида s3://bucket/path")
        if (self.shard_id_column is None) != (self.num_shards is None):
            raise ValueError("shard_id_column и num_shards нужно передать вместе")
        suffixes = [""]
        if self.shard_id_column is not None:
            if not self.shard_id_column or any(c in self.shard_id_column for c in "/=?#"):
                raise ValueError("shard_id_column должен содержать имя колонки")
            count = int(self.num_shards)
            if isinstance(self.num_shards, bool) or str(count) != str(self.num_shards) or count < 1:
                raise ValueError("num_shards должен быть положительным целым числом")
            suffixes = [f"/{self.shard_id_column}={i}" for i in range(count)]
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
                    "calc_utilization": self.calc_utilization,
                    "s3_input_path": self.s3_input_prefix.rstrip("/") + suffix,
                    "s3_output_path": self.s3_output_prefix.rstrip("/") + suffix,
                }
            )
        return requests

    def _http(self, method, path, payload=None):
        """
        Повторяет временные HTTP-ошибки с тем же телом и ключом запроса.
        """
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        tls_context = ssl.create_default_context()
        if not self.verify_ssl:
            tls_context.check_hostname = False
            tls_context.verify_mode = ssl.CERT_NONE
        for attempt in range(3):
            try:
                request = Request(
                    self.service_url.rstrip("/") + path,
                    data=data,
                    method=method,
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(
                    request, timeout=self.request_timeout, context=tls_context
                ) as response:
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
        """
        Отправляет отмену всем известным незавершённым задачам, даже при ошибках API.
        """
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
        """
        Восстанавливает заявки по ключам и проверяет завершение всех шардов.
        """
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
                progress = task.get("progress_pct")
                eta = task.get("eta_seconds")
                progress_text = "нет данных" if progress is None else f"{progress:.1f}%"
                eta_text = "нет оценки" if eta is None else f"{eta:.0f} с"
                self.log.info(
                    f"Инференс {task_id}: {status}, прогресс={progress_text}, "
                    f"строк={task.get('processed_rows', '?')}/{task.get('total_rows', '?')}, "
                    f"ETA={eta_text}, выход={payload['s3_output_path']}"
                )
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
        """
        Запрашивает отмену при штатном завершении активного процесса Airflow.
        """
        self._abort_remaining()
