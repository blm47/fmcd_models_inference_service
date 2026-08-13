"""
Обёртка над S3 для потокового чтения/записи PARQUET-ПРЕФИКСОВ (не одиночных
файлов).

Контракт входа/выхода: s3_input_path и s3_output_path — это ПРЕФИКСЫ
(папки), под которыми лежит/будет лежать множество parquet-файлов:
  s3://bucket/path/input/
    _SUCCESS
    part-00000-....snappy.parquet
    part-00001-....snappy.parquet
    ...
Именно так Spark сохраняет DataFrame через df.write.parquet(prefix).

ВАЖНО про S3-бэкенд: используется s3fs.S3FileSystem, а НЕ
pyarrow.fs.S3FileSystem. Причина — у pyarrow.fs.S3FileSystem нет
официального способа отключить проверку TLS-сертификата (актуально для
внутренних S3-эндпоинтов с самоподписанным/корпоративным сертификатом,
см. verify_ssl в S3Config), тогда как s3fs/boto3 поддерживают это
штатно через client_kwargs={"verify": False}.

pyarrow.dataset.dataset(...) умеет принимать fsspec-совместимую
файловую систему (s3fs реализует fsspec.AbstractFileSystem), поэтому
весь остальной код (open_dataset/iter_chunks/writer) не меняется.

Правильный путь для прода — не отключать verify_ssl, а подложить
корпоративный CA-сертификат в образ (update-ca-certificates в
Dockerfile) и держать verify_ssl=True. verify_ssl=False — временный
обход для тестового стенда без валидного сертификата.
"""

from dataclasses import dataclass, field

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import s3fs

from app.core.config import S3Config

_SUCCESS_MARKER = "_SUCCESS"


@dataclass
class S3Client:
    settings: S3Config

    def _filesystem(self) -> s3fs.S3FileSystem:
        client_kwargs = {}
        if not self.settings.verify_ssl:
            client_kwargs["verify"] = False

        return s3fs.S3FileSystem(
            key=self.settings.access_key,
            secret=self.settings.secret_key,
            client_kwargs={
                "endpoint_url": self.settings.endpoint_url,
                "region_name": self.settings.region,
                **client_kwargs,
            },
            use_ssl=self.settings.use_ssl,
        )

    @staticmethod
    def _strip_bucket_prefix(s3_path: str) -> str:
        # pyarrow.dataset ожидает путь без "s3://" при явно переданной filesystem
        return s3_path.rstrip("/").replace("s3://", "", 1)

    def open_dataset(self, s3_prefix: str) -> ds.Dataset:
        """
        Открывает ВСЕ *.parquet файлы под префиксом как единый dataset.
        _SUCCESS и прочие не-parquet файлы pyarrow.dataset игнорирует
        автоматически (format="parquet" фильтрует по расширению .parquet).
        """
        fs = self._filesystem()
        prefix = self._strip_bucket_prefix(s3_prefix)
        return ds.dataset(prefix, filesystem=fs, format="parquet")

    def count_rows(self, s3_prefix: str) -> int:
        """Дешёвый подсчёт строк по метаданным всех part-файлов, без чтения данных."""
        dataset = self.open_dataset(s3_prefix)
        return dataset.count_rows()

    def get_schema_columns(self, s3_prefix: str) -> set[str]:
        dataset = self.open_dataset(s3_prefix)
        return set(dataset.schema.names)

    def iter_chunks(self, s3_prefix: str, chunk_size: int):
        """
        Генератор df_chunk по chunk_size строк. pyarrow.dataset.to_batches
        сам "склеивает" record batches из разных part-файлов Spark в батчи
        нужного размера, поэтому границы chunk_size не привязаны к границам
        исходных part-файлов.
        """
        dataset = self.open_dataset(s3_prefix)
        for batch in dataset.to_batches(batch_size=chunk_size):
            yield batch.to_pandas()

    def get_writer(self, s3_output_prefix: str) -> "ParquetPrefixWriter":
        """
        Возвращает писатель, который на каждый write_chunk создаёт новый
        part-файл под выходным префиксом (по аналогии со Spark part-файлами),
        а на close() кладёт пустой _SUCCESS маркер.
        """
        return ParquetPrefixWriter(
            fs=self._filesystem(),
            output_prefix=self._strip_bucket_prefix(s3_output_prefix),
        )


@dataclass
class ParquetPrefixWriter:
    fs: s3fs.S3FileSystem
    output_prefix: str
    _part_idx: int = field(default=0, init=False)

    def write_chunk(self, df_chunk: pd.DataFrame) -> None:
        """Пишет один чанк как отдельный part-файл под output_prefix."""
        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
        part_path = f"{self.output_prefix}/part-{self._part_idx:05d}.parquet"
        with self.fs.open(part_path, "wb") as sink:
            pq.write_table(table, sink)
        self._part_idx += 1

    def close(self) -> None:
        """Кладёт пустой _SUCCESS маркер - признак полностью завершённой записи."""
        success_path = f"{self.output_prefix}/{_SUCCESS_MARKER}"
        with self.fs.open(success_path, "wb") as sink:
            sink.write(b"")
