import argparse
import base64
import datetime as dt
import json
import re
import uuid
import zlib
from contextlib import closing
from copy import deepcopy
from pathlib import Path


def as_bool(value, name):
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"{name} должен быть булевым значением")


def positive_int(value, name):
    if not re.fullmatch(r"[0-9]+", str(value).strip()) or int(value) < 1:
        raise ValueError(f"{name} должен быть положительным целым числом")
    return int(value)


def column_name(value, name):
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} должен содержать непустое имя колонки")
    return value


def normalize_args(raw):
    """
    Проверяет настройки записи и шардирования до запуска расчёта.
    """
    defaults = {
        "shard_column": None,
        "num_shards": None,
        "shard_id_column": "shard_id",
        "partition_by": None,
        "repartition": 1,
        "safe_upload": False,
        "clear_s3_path": False,
        "add_load_id": True,
        "use_parquet_for_meta": True,
        "connection_ssl": True,
        "use_bulk_committer": True,
        "no_delete_meta_file": False,
        "dt_formatter": "%Y%m%d%H%M%S",
        "mode": "overwrite",
    }
    args = {**defaults, **deepcopy(raw)}

    # Airflow может передать шаблонизированные bool и int строками.
    for name, default in defaults.items():
        if isinstance(default, bool):
            args[name] = as_bool(args[name], name)

    enabled = args["shard_column"] is not None or args["num_shards"] is not None
    if enabled:
        args["shard_column"] = column_name(args["shard_column"], "shard_column")
        args["num_shards"] = positive_int(args["num_shards"], "num_shards")
        args["shard_id_column"] = column_name(args["shard_id_column"], "shard_id_column")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args["shard_id_column"]):
            raise ValueError("shard_id_column должен быть простым SQL-идентификатором")
        
    parts = args["partition_by"]
    parts = [parts] if isinstance(parts, str) else list(parts or [])
    parts = [column_name(part, "partition_by") for part in parts]

    if enabled and args["shard_id_column"] not in parts:
        parts.append(args["shard_id_column"])
    args["partition_by"] = parts
    count = args["repartition"]
    args["repartition"] = positive_int(count if count is not None else 1, "repartition")

    if args["mode"] not in ("overwrite", "append", "ignore", "error", "errorifexists"):
        raise ValueError("Неподдерживаемый режим записи")
    if args["mode"] == "ignore" and args["clear_s3_path"]:
        raise ValueError("clear_s3_path несовместим с mode=ignore")
    return args


def decode_payload(encoded):
    """
    Декодирует результат application_args_encoding, включая обрамляющие кавычки.
    """
    text = encoded.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1]
    raw = json.loads(zlib.decompress(base64.b64decode(text, validate=True)).decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Закодированное значение должно содержать JSON-объект")
    return raw


def wrap_query(sql, shard_column, num_shards, shard_id_column="shard_id"):
    """
    Оборачивает один SELECT после Jinja, сохраняя CTE, комментарии и литералы.
    """
    import sqlparse
    from sqlparse import tokens as T

    statements = [
        s
        for s in sqlparse.parse(sql)
        if any(
            not t.is_whitespace and t.ttype not in T.Comment and t.value != ";" for t in s.flatten()
        )
    ]
    if len(statements) != 1 or statements[0].get_type() != "SELECT":
        raise ValueError("Для шардирования нужен один SELECT, допускается WITH ... SELECT")
    
    # Удаляем разделитель statement, не затрагивая ';' внутри строк и комментариев.
    query = "".join(
        t.value
        for t in statements[0].flatten()
        if not (t.ttype in T.Punctuation and t.value == ";")
    ).strip()
    count = positive_int(num_shards, "num_shards")
    key = column_name(shard_column, "shard_column").replace("`", "``")
    shard = column_name(shard_id_column, "shard_id_column").replace("`", "``")
    return (
        "SELECT\n    _shard_source.*\n"
        f"    , pmod(xxhash64(_shard_source.`{key}`), {count}) AS `{shard}`\n"
        f"FROM (\n{query}\n) AS _shard_source"
    )


def validate_columns(columns, args, case_sensitive=False):
    """
    Проверяет наличие и однозначность колонок перед записью parquet.
    """
    norm = (lambda value: value) if case_sensitive else str.casefold
    names = [norm(name) for name in columns]
    if len(names) != len(set(names)):
        raise ValueError("Результат SELECT содержит неоднозначные имена колонок")
    parts = [norm(name) for name in args["partition_by"]]
    if len(parts) != len(set(parts)):
        raise ValueError("partition_by содержит неоднозначные имена колонок")
    if any(name not in names for name in parts):
        raise ValueError("В результате нет колонки из partition_by")
    if parts and len(parts) == len(names):
        raise ValueError("Для parquet нужна хотя бы одна колонка вне partition_by")


def destination(args, metadata=False):
    """
    Строит путь данных или метаданных с учётом load_id.
    """
    bucket = args["s3_bucket"]
    key = args["s3_meta_path" if metadata else "s3_path"].strip("/")
    if not bucket or "/" in bucket or not key or "://" in key:
        raise ValueError("Нужны имя бакета и непустой относительный путь S3")
    if any(part in ("", ".", "..") for part in key.split("/")):
        raise ValueError("Путь S3 содержит пустой или относительный компонент")
    if args["add_load_id"]:
        load_id = str(args["load_id"])
        if not load_id or any(char in load_id for char in "/?#") or load_id in (".", ".."):
            raise ValueError("load_id должен быть одним непустым компонентом пути")
        key += "/load_id=" + load_id
    return "s3a://" + bucket + "/" + key


def build_spark_conf(args):
    """
    Настраивает S3A и committer для текущего режима выгрузки.
    """
    from pyspark import SparkConf

    conf = SparkConf()
    properties = {
        "fs.s3a.endpoint": args["endpoint"],
        "fs.s3a.attempts.maximum": args["attempts_maximum"],
        "fs.s3a.retry.interval": args["retry_interval"],
        "fs.s3a.connection.ssl.enabled": str(args["connection_ssl"]).lower(),
        "fs.s3a.connection.timeout": positive_int(args["connection_timeout"], "connection_timeout")
        * 1000,
        "fs.s3a.connection.ttl": args["connection_ttl"],
        "fs.s3a.path.style.access": "true",
        "fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
    }
    if args.get("access_key"):
        properties["fs.s3a.access.key"] = args["access_key"]
    if args["connection_ssl"]:
        properties["fs.s3a.ssl.channel.mode"] = "Default_JSSE_with_GCM"
    if args.get("keystore_location") and args.get("truststore_location"):
        properties.update(
            {
                "fs.s3a.connection.ssl.keystore.type": "PKCS12",
                "fs.s3a.connection.ssl.keystore.location": args["keystore_location"],
                "fs.s3a.connection.ssl.truststore.location": args["truststore_location"],
            }
        )
    if args["use_bulk_committer"]:
        # Сохраняем настройки committer, используемые в окружении Hadoop.
        properties.update(
            {
                "fs.s3a.committer.name": "directory",
                "fs.s3a.committer.magic.enable": "true",
                "fs.s3a.committer.staging.tmp.path": "hdfs:///tmp/s3a_staging",
                "fs.s3a.committer.staging.conflict-mode": "append",
                "fs.s3a.fast.upload": "true",
                "mapreduce.fileoutputcommitter.algorithm.version": "2",
                "fs.s3a.multipart.enable": "false",
                "fs.s3a.multipart.size": str(50 * 1024**3),
                "fs.s3a.multipart.threshold": str(50 * 1024**3),
                "fs.s3a.committer.upload.mode": "single",
            }
        )
    if args["safe_upload"]:
        properties.update(
            {
                "fs.s3a.connection.maximum": "4",
                "fs.s3a.threads.max": "2",
                "fs.s3a.max.total.tasks": "2",
                "fs.s3a.attempts.maximum": "20",
            }
        )
    for name, value in properties.items():
        conf.set("spark.hadoop." + name, str(value))
    return conf


def write_data(spark, frame, args, path):
    """
    Записывает данные целиком или по колонкам partition_by.
    """
    if args["clear_s3_path"]:
        java_path = spark._jvm.org.apache.hadoop.fs.Path(path)
        fs = java_path.getFileSystem(spark.sparkContext._jsc.hadoopConfiguration())
        if fs.exists(java_path) and not fs.delete(java_path, True):
            raise RuntimeError("Не удалось очистить указанный выходной префикс")
    writer = frame.repartition(args["repartition"]).write.mode(args["mode"])
    if args["partition_by"]:
        writer = writer.partitionBy(*args["partition_by"])
    writer.parquet(path)


def append_csv_boto(spark, args, metadata, logger):
    """
    Дописывает строку в CSV через GET и PUT, не выполняя DELETE в S3.
    """
    import boto3
    import pandas as pd
    from botocore.config import Config
    from botocore.exceptions import ClientError

    credentials = decode_payload(spark.sparkContext.getConf().get("spark.app.oper.s3_password"))

    # Это контракт библиотеки: здесь удаляются все '/', а load_id не добавляется.
    key = f"{args['s3_meta_path'].replace('/', '')}/{args['object_name'].split('.')[-1]}.csv"
    client = boto3.client(
        "s3",
        endpoint_url=args["endpoint"],
        aws_access_key_id=credentials["access_key"],
        aws_secret_access_key=credentials["secret_key"],
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        use_ssl=True,
        verify=False,
    )
    with closing(client):
        try:
            response = client.get_object(Bucket=args["s3_bucket"], Key=key)
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in {"404", "NoSuchKey"}:
                raise
            previous = pd.DataFrame(columns=metadata.columns)
        else:
            with closing(response["Body"]) as body:
                previous = pd.read_csv(body, usecols=metadata.columns)
        combined = pd.concat([previous, metadata], ignore_index=True)
        client.put_object(Bucket=args["s3_bucket"], Key=key, Body=combined.to_csv(index=False))
        logger.info(f"Метаданные записаны: s3://{args['s3_bucket']}/{key}")


def append_csv_spark(spark, args, metadata, path, header, logger):
    """
    Обновляет именованный CSV через Spark, материализуя результат до удаления оригинала.
    """
    hadoop_path = spark._jvm.org.apache.hadoop.fs.Path
    fs = hadoop_path(path).getFileSystem(spark.sparkContext._jsc.hadoopConfiguration())
    target = hadoop_path(f"{path}/{args['object_name'].split('.')[-1]}.csv")
    result = metadata
    if fs.exists(target):
        previous = spark.read.schema(metadata.schema).csv(target.toString(), header=header)
        result = previous.unionByName(metadata)

    # Старый CSV остаётся доступным, пока Spark читает его и пишет объединённый результат.
    staging = hadoop_path(f"{path}/_metadata_staging_{uuid.uuid4().hex}")
    try:
        result.coalesce(1).write.mode("errorifexists").option("header", header).csv(
            staging.toString()
        )
        parts = [
            item.getPath()
            for item in fs.listStatus(staging)
            if item.isFile() and item.getPath().getName().startswith("part-")
        ]
        if len(parts) != 1:
            raise RuntimeError("Ожидался один CSV part-файл метаданных")
        if fs.exists(target) and not fs.delete(target, False):
            raise RuntimeError("Не удалось заменить старый CSV метаданных")
        if not fs.rename(parts[0], target):
            raise RuntimeError(f"Не удалось опубликовать CSV; результат сохранён в {staging}")
    except Exception:
        # При ошибке оставляем временный результат для восстановления.
        logger.error(f"Ошибка публикации CSV, проверьте временный путь: {staging}")
        raise
    else:
        if not fs.delete(staging, True):
            logger.warn(f"Не удалось удалить временный путь метаданных: {staging}")
    logger.info(f"Метаданные записаны: {target}")


def write_metadata(spark, frame, args, path, logger):
    """
    Рассчитывает метаданные и выбирает запись parquet или именованного CSV.
    """
    import pandas as pd

    meta = deepcopy(args.get("meta_variables") or {})
    if not meta:
        return
    
    header = as_bool(meta.pop("header", False), "header")
    calc_meta = {
        **(args.get("calc_meta") or {}),
        "object_name": args.get("object_name"),
        "load_id": args["load_id"],
        "start_time": args["load_start_time"],
    }

    # Не запускаем дополнительный Spark action, если метрика не запрошена.
    if meta.get("inc_rowcount") == "None":
        calc_meta["inc_rowcount"] = frame.count()

    calc_meta["finish_time"] = dt.datetime.now().strftime(args["dt_formatter"])
    for key, value in calc_meta.items():
        if key in meta and meta[key] == "None":
            meta[key] = value
    if not meta:
        return
    
    metadata_frame = pd.DataFrame({key: [value] for key, value in meta.items()})
    if not args["use_parquet_for_meta"] and args["no_delete_meta_file"]:
        append_csv_boto(spark, args, metadata_frame, logger)
        return
    
    metadata = spark.createDataFrame(metadata_frame)
    if args["use_parquet_for_meta"]:
        metadata.coalesce(1).write.mode(args["mode"]).option("header", header).parquet(path)
        logger.info(f"Метаданные записаны: {path}")
    else:
        append_csv_spark(spark, args, metadata, path, header, logger)


def read_sql_file(path):
    """
    Читает SQL, доставленный Spark через --files, или локальный файл.
    """
    from pyspark import SparkFiles

    distributed = Path(SparkFiles.get(Path(path).name))
    if distributed.is_file():
        return distributed.read_text(encoding="utf-8")
    return Path(path).read_text(encoding="utf-8")


def run(spark, args, logger):
    """
    Выполняет запрос и выбранную фазу выгрузки.
    """
    from jinja2 import Template

    if not args.get("query_path"):
        raise ValueError("Нужно передать query_path")
    
    data_path = destination(args)
    meta_path = destination(args, metadata=True) if args.get("meta_variables") else None

    if meta_path and (
        (data_path + "/").startswith(meta_path + "/")
        or (meta_path + "/").startswith(data_path + "/")
    ):
        raise ValueError("Пути данных и метаданных не должны пересекаться")
    if meta_path and not args["use_parquet_for_meta"] and not args.get("object_name"):
        raise ValueError("Для именованного CSV нужно передать object_name")
    sql = Template(read_sql_file(args["query_path"])).render(args.get("template_variables") or {})
    if args["use_bulk_committer"] and args["num_shards"] is not None:
        sql = wrap_query(sql, args["shard_column"], args["num_shards"], args["shard_id_column"])
        
    # Spark проверяет ссылки на колонки; дубли в итоговой схеме проверяем до очистки S3.
    frame = spark.sql(sql)
    if args["use_bulk_committer"]:
        validate_columns(
            frame.columns,
            args,
            spark.conf.get("spark.sql.caseSensitive", "false").lower() == "true",
        )
    if frame.rdd.isEmpty():
        logger.info("Данных во входной таблице нет; запись пропущена")
        return
    if args["use_bulk_committer"]:
        write_data(spark, frame, args, data_path)
        logger.info("Данные записаны: " + data_path)
    else:
        # Метаданные не шардируются, эта фаза не очищает путь данных.
        write_metadata(spark, frame, args, meta_path, logger)


def main():
    """
    Создаёт SparkSession и единый logger, освобождает Spark после выполнения.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--application_args_encoded", required=True)
    args = normalize_args(decode_payload(parser.parse_args().application_args_encoded))
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.config(conf=build_spark_conf(args)).enableHiveSupport().getOrCreate()
    )
    logger = initialize_logger(spark)
    try:
        # Используем установленный в окружении S3A-коннектор.
        spark._jvm.org.apache.hadoop.fs.FileSystem.getFileSystemClass(
            "s3a", spark.sparkContext._jsc.hadoopConfiguration()
        )
        run(spark, args, logger)
    finally:
        spark.stop()


def initialize_logger(spark):
    """
    Получает единый logger с форматом и appenders, настроенными окружением Spark.
    """
    return spark._jvm.org.apache.log4j.LogManager.getLogger("HadoopToS3")


if __name__ == "__main__":
    main()
