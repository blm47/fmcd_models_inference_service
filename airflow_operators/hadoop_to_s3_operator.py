import datetime as dt
import os
import pathlib
import sys
from copy import deepcopy

try:
    from airflow.contrib.operators.spark_submit_operator import SparkSubmitOperator
    from airflow.hooks.base_hook import BaseHook
except ImportError:
    from airflow.hooks.base import BaseHook
    from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils import timezone

sys.path.append(os.path.join(f"{pathlib.Path(__file__).parent.parent.resolve()}/utils"))
from general_utils import application_args_encoding

_DEFAULT_METRICS = object()


class HadoopToS3Operator(SparkSubmitOperator):
    """
    Запускает выгрузку данных и затем отдельный Spark job для метаданных.
    """

    template_fields = tuple(
        dict.fromkeys(
            (
                "_endpoint",
                "_connection_name",
                "_access_key",
                "_secret_key",
                "_s3_bucket",
                "_object_name",
                "_query_path",
                "_template_variables",
                "_s3_path",
                "_s3_meta_path",
                "_use_parquet_for_meta",
                "_add_load_id",
                "_clear_s3_path",
                "_repartition",
                "_mode",
                "_connection_ssl",
                "_attempts_maximum",
                "_retry_interval",
                "_connection_timeout",
                "_connection_ttl",
                "_keystore_location",
                "_truststore_location",
                "_meta_variables",
                "_dt_formatter",
                "_files",
                "_shard_column",
                "_num_shards",
                "_shard_id_column",
                "_partition_by",
                "_load_id",
                *SparkSubmitOperator.template_fields,
            )
        )
    )

    def __init__(
        self,
        query_path: str = None,
        object_name: str = None,
        endpoint: str = None,
        access_key: str = None,
        secret_key: str = None,
        s3_bucket: str = None,
        connection_name: str = None,
        keystore_location: str = None,
        keystore_password: str = None,
        truststore_location: str = None,
        truststore_password: str = None,
        template_variables: dict = None,
        s3_path: str = "from_dapp/model_result",
        s3_meta_path: str = "from_dapp/meta/model_result",
        use_parquet_for_meta: bool = True,
        no_delete_meta_file: bool = False,
        add_load_id: bool = True,
        clear_s3_path: bool = False,
        repartition: int = 1,
        safe_upload: bool = False,
        mode: str = "overwrite",
        connection_ssl: bool = True,
        attempts_maximum: int = 1,
        retry_interval: int = 60,
        connection_timeout: int = 300,
        connection_ttl: int = 300,
        dt_formatter: str = "%Y%m%d%H%M%S",
        metrics_calc_variables=_DEFAULT_METRICS,
        shard_column: str = None,
        num_shards: int = None,
        shard_id_column: str = "shard_id",
        partition_by=None,
        load_id: str = None,
        count_loaded_keys: bool = True,
        **kwargs,
    ) -> None:
        """
        Сохраняет аргументы; значения шаблонов проверяются после рендеринга.

        :param query_path: Путь к файлу SQL, который Spark получает через --files.
        :param object_name: Наименование витрины для метаданных и имени CSV-файла.
        :param endpoint: URL S3, если подключение задано без connection_name.
        :param access_key: Access key для подключения к S3.
        :param secret_key: Secret key для подключения к S3.
        :param s3_bucket: Имя бакета S3 для записи данных и метаданных.
        :param connection_name: Airflow connection с ключами в login/password
            и endpoint_url, bucket_name в extra.service_config.s3.
            Если задан, настройки подключения берутся из него.
        :param keystore_location: Путь к mTLS keystore в формате PKCS12.
        :param keystore_password: Пароль mTLS keystore.
        :param truststore_location: Путь к mTLS truststore.
        :param truststore_password: Пароль mTLS truststore.
        :param template_variables: Переменные для Jinja-подстановки в исходном SQL.
        :param s3_path: Путь данных внутри бакета; по умолчанию from_dapp/model_result.
        :param s3_meta_path: Путь метаданных; по умолчанию from_dapp/meta/model_result.
        :param use_parquet_for_meta: True — метаданные в Parquet, False — в CSV.
        :param no_delete_meta_file: Для CSV добавлять строки через boto3 GET/PUT
            без DELETE. Путь сохраняет исходный формат библиотеки:
            s3_meta_path без символов '/' + '/' + короткое имя object_name + '.csv',
            без load_id. При False используется запись CSV через Spark.
        :param add_load_id: Добавлять /load_id=<load_id> к путям данных и метаданных,
            кроме CSV при no_delete_meta_file=True. По умолчанию True.
        :param clear_s3_path: Очищать текущий путь данных перед записью.
            Фаза метаданных данные не очищает. По умолчанию False.
        :param repartition: Число Spark-партиций перед записью, по умолчанию 1.
            Независимо от num_shards; внутри шарда может быть несколько Parquet-файлов.
        :param safe_upload: Ограничивать число соединений и потоков загрузки в S3.
        :param mode: Режим записи Spark, по умолчанию overwrite; также поддержаны
            append, ignore, error и errorifexists. Overwrite заменяет текущий путь
            целиком, append добавляет файлы. Накопление CSV имеет отдельную логику.
        :param connection_ssl: Использовать SSL для S3A, по умолчанию True.
        :param attempts_maximum: Максимальное число попыток S3A, по умолчанию 1.
        :param retry_interval: Интервал повторных попыток S3A, по умолчанию 60.
        :param connection_timeout: Таймаут подключения в секундах, по умолчанию 300.
        :param connection_ttl: Время жизни соединения S3A, по умолчанию 300.
        :param dt_formatter: Формат времени метаданных, по умолчанию %Y%m%d%H%M%S.
        :param metrics_calc_variables: Поля метаданных. Строка 'None' запрашивает
            расчёт поддерживаемого поля; явные значения сохраняются. Если аргумент
            опущен, используется стандартный набор полей. None или {} отключают
            второй Spark job. Опечатка mertics_calc_variables поддержана через kwargs.
        :param shard_column: Колонка результата SQL для xxhash64, например customer_mdm_id.
            Задаётся вместе с num_shards; по умолчанию шардирование выключено.
        :param num_shards: Положительное число шардов. Номер вычисляется после Jinja
            как pmod(xxhash64(shard_column), num_shards), без изменения типа ключа.
        :param shard_id_column: Имя добавляемой колонки и каталогов, по умолчанию
            shard_id. Не должно совпадать с колонкой результата SQL.
        :param partition_by: Дополнительная колонка или список колонок каталогов.
            При шардировании shard_id_column автоматически добавляется в конец,
            если её ещё нет в списке. По умолчанию дополнительных партиций нет.
        :param load_id: Общий идентификатор загрузки данных и метаданных.
            Если не задан, создаётся из текущего времени в формате %Y%m%d%H%M%S.
        :param count_loaded_keys: Считать Parquet под текущим путём загрузки
            и публиковать cnt_loaded_keys в XCom. По умолчанию True;
            при ошибке подсчёта публикуется -1.
        :param kwargs: Остальные параметры SparkSubmitOperator, включая task_id и conf.
        """

        if not (connection_name or (endpoint and access_key and secret_key and s3_bucket)):
            raise ValueError(
                "Не указан connection_name или endpoint, access_key, secret_key, s3_bucket"
            )
        
        # Сохраняем совместимость с опечаткой в старых DAG исходной библиотеки.
        if "mertics_calc_variables" in kwargs:
            metrics_calc_variables = kwargs.pop("mertics_calc_variables")

        super().__init__(**kwargs)
        self._connection_name = connection_name
        self._endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._s3_bucket = s3_bucket
        self._object_name = object_name
        self._query_path = query_path and pathlib.Path(query_path).name
        self._template_variables = deepcopy(template_variables or {})
        self._s3_path = s3_path
        self._s3_meta_path = s3_meta_path
        self._use_parquet_for_meta = use_parquet_for_meta
        self._no_delete_meta_file = no_delete_meta_file
        self._add_load_id = add_load_id
        self._clear_s3_path = clear_s3_path
        self._repartition = repartition
        self._safe_upload = safe_upload
        self._mode = mode
        self._connection_ssl = connection_ssl
        self._attempts_maximum = attempts_maximum
        self._retry_interval = retry_interval
        self._connection_timeout = connection_timeout
        self._connection_ttl = connection_ttl
        self._keystore_location = keystore_location
        self._keystore_password = keystore_password
        self._truststore_location = truststore_location
        self._truststore_password = truststore_password
        self._dt_formatter = dt_formatter
        self._shard_column = shard_column
        self._num_shards = num_shards
        self._shard_id_column = shard_id_column
        self._partition_by = deepcopy(partition_by)
        self._load_id = load_id
        self._count_loaded_keys = count_loaded_keys

        if metrics_calc_variables is _DEFAULT_METRICS:
            metrics_calc_variables = dict.fromkeys(
                (
                    "object_name",
                    "load_id",
                    "inc_rowcount",
                    "increment_field_name",
                    "increment_from_value",
                    "increment_to_value",
                    "start_time_sumk",
                    "finish_time_sumk",
                    "status_sumk",
                ),
                "None",
            )
        self._meta_variables = deepcopy(metrics_calc_variables)
        self._files = query_path
        self._application = str(
            pathlib.Path(__file__).parent.resolve() / "scripts" / "hadoop_to_s3_script.py"
        )

    def _run_spark_job(self, use_bulk_committer, context):
        """
        Передаёт аргументы через существующий encoder библиотеки.
        """
        application_args = deepcopy(self._application_args_dict)
        application_args["use_bulk_committer"] = use_bulk_committer
        if not use_bulk_committer:
            application_args["clear_s3_path"] = False
        encoded = application_args_encoding(application_args)
        self._application_args = [f"--application_args_encoded='{encoded}'"]
        self._conf = dict(self._conf or {})
        self._conf["spark.hadoop.fs.s3a.secret.key"] = self._secret_key
        if self._keystore_location and self._truststore_location:
            self._conf.update(
                {
                    "spark.hadoop.fs.s3a.connection.ssl.keystore.password": self._keystore_password,
                    "spark.hadoop.fs.s3a.connection.ssl.truststore.password": self._truststore_password,
                }
            )
        self._conf["spark.app.oper.s3_password"] = application_args_encoding(
            {
                "access_key": self._access_key,
                "secret_key": self._secret_key,
            }
        )
        super().execute(context)

    def execute(self, context):
        """
        Загружает данные и метаданные с общим load_id, публикует XCom.
        """
        calc_meta = {}
        if self._connection_name:
            connection = BaseHook.get_connection(self._connection_name)
            try:
                service = connection.extra_dejson["service_config"]
                self._endpoint = service["s3"]["endpoint_url"]
                self._s3_bucket = service["s3"]["bucket_name"]
                self._access_key = connection.login
                self._secret_key = connection.password
                calc_meta["tuz_loader"] = service["s3"].get("tuz_loader")
            except (KeyError, TypeError):
                self.log.error(
                    "Некорректный extra в Airflow connection: ожидается service_config.s3"
                )
                raise
        fields = (
            "endpoint",
            "s3_bucket",
            "access_key",
            "truststore_location",
            "keystore_location",
            "object_name",
            "query_path",
            "template_variables",
            "s3_path",
            "s3_meta_path",
            "use_parquet_for_meta",
            "no_delete_meta_file",
            "add_load_id",
            "clear_s3_path",
            "repartition",
            "safe_upload",
            "mode",
            "connection_ssl",
            "attempts_maximum",
            "retry_interval",
            "connection_timeout",
            "connection_ttl",
            "dt_formatter",
            "shard_column",
            "num_shards",
            "shard_id_column",
            "partition_by",
        )
        self._application_args_dict = {name: deepcopy(getattr(self, "_" + name)) for name in fields}
        self._application_args_dict.update(
            {
                "meta_variables": deepcopy(self._meta_variables),
                "calc_meta": calc_meta,
                "load_id": self._load_id or dt.datetime.now().strftime("%Y%m%d%H%M%S"),
                "load_start_time": dt.datetime.now().strftime(self._dt_formatter),
            }
        )
        
        context["ti"].xcom_push(key="mark_loaded_at", value=timezone.utcnow().isoformat())
        self.log.info("Запуск загрузки данных")
        self._run_spark_job(True, context)
        if self._meta_variables:
            self.log.info("Запуск загрузки метаданных")
            self._hook = None
            self._run_spark_job(False, context)
        if self._count_loaded_keys:
            self._count_parquet(context)

    def _count_parquet(self, context):
        """
        Считает parquet текущего пути загрузки, включая вложенные шарды.
        """
        client = None
        try:
            import boto3
            from botocore.client import Config as BotoConfig

            client = boto3.client(
                "s3",
                endpoint_url=self._endpoint,
                aws_access_key_id=self._access_key,
                aws_secret_access_key=self._secret_key,
                config=BotoConfig(signature_version="s3v4", s3={"addressing_style": "path"}),
                use_ssl=True,
                verify=False,
            )
            prefix = self._s3_path.rstrip("/")
            add_load_id = self._add_load_id
            if isinstance(add_load_id, str):
                add_load_id = add_load_id.strip().lower() in ("true", "1", "yes", "on")
            if add_load_id:
                prefix += "/load_id=" + str(self._application_args_dict["load_id"])
            count = sum(
                key.endswith(".parquet")
                for key in self._list_keys(client, self._s3_bucket, prefix + "/")
            )
        except Exception:
            count = -1
            self.log.warn("Не удалось подсчитать parquet в S3 после загрузки")
        finally:
            if client is not None:
                client.close()
        context["ti"].xcom_push(key="cnt_loaded_keys", value=count)

    def _list_keys(self, s3_client, bucket: str, prefix: str):
        """
        Возвращает ключи под prefix с учётом пагинации S3.
        """
        return [
            item["Key"]
            for page in s3_client.get_paginator("list_objects_v2").paginate(
                Bucket=bucket, Prefix=prefix
            )
            for item in page.get("Contents", [])
        ]
