"""
Выбор backend по YAML и подключение по ENV.
"""

import boto3
from botocore.config import Config

from app.tasks.backends.s3 import S3TaskBackend


def create_task_backend(settings, logger):
    queue = settings.task_store
    if queue.backend == "pg":
        from app.tasks.backends.postgres import PostgresTaskBackend

        if settings.postgres is None:
            raise ValueError("Не заданы параметры подключения PostgreSQL")
        return PostgresTaskBackend(settings.postgres, queue, logger)
    s3 = settings.s3
    client = boto3.client(
        "s3",
        endpoint_url=s3.endpoint_url,
        aws_access_key_id=s3.access_key,
        aws_secret_access_key=s3.secret_key,
        region_name=s3.region,
        use_ssl=s3.use_ssl,
        verify=s3.verify_ssl,
        config=Config(
            signature_version="s3v4",
            connect_timeout=queue.connect_timeout_sec,
            read_timeout=queue.read_timeout_sec,
            retries={"mode": "standard", "total_max_attempts": 1},
        ),
    )
    return S3TaskBackend(client, s3.bucket_out, queue, logger)
