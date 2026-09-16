"""
Потоковое чтение parquet из S3 и запись частей результата с маркером _SUCCESS.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import s3fs

from app.core.config import S3Config
from app.storage.row_group_check import warn_on_oversized_row_groups

_SUCCESS_MARKER = "_SUCCESS"


@dataclass
class ParquetPrefixWriter:
    fs: s3fs.S3FileSystem
    output_prefix: str
    _part_idx: int = field(default=0, init=False)
    _schema: pa.Schema | None = field(default=None, init=False)
    _failed: bool = field(default=False, init=False)

    def write_chunk(self, df_chunk: pd.DataFrame) -> None:
        """
        Пишет один чанк как отдельный парт-файл под output_prefix.
        """
        if self._failed:
            raise RuntimeError("Writer остановлен после ошибки схемы результата")
        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
        schema = table.schema.remove_metadata()
        if self._schema is not None and not self._schema.equals(schema):
            self._failed = True
            raise ValueError("Схема Parquet результата изменилась между чанками")
        self._schema = schema
        part_path = f"{self.output_prefix}/part-{self._part_idx:05d}.parquet"
        with self.fs.open(part_path, "wb") as sink:
            pq.write_table(table, sink)
        self._part_idx += 1

    def close(self) -> None:
        """
        Кладёт пустой _SUCCESS маркер - признак полностью завершённой записи.
        """
        if self._failed:
            raise RuntimeError("Нельзя публиковать _SUCCESS после ошибки схемы результата")
        success_path = f"{self.output_prefix}/{_SUCCESS_MARKER}"
        with self.fs.open(success_path, "wb") as sink:
            sink.write(b"")


@dataclass
class S3Client:
    settings: S3Config
    logger: Any

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
            skip_instance_cache=True,  # не переиспользовать закэшированный инстанс
            use_listings_cache=False,  # не кэшировать листинги директорий вовсе
        )

    @staticmethod
    def _strip_bucket_prefix(s3_path: str) -> str:
        # pyarrow.dataset ожидает путь без "s3://"
        return s3_path.rstrip("/").replace("s3://", "", 1)

    def open_dataset(self, s3_prefix: str) -> ds.Dataset:
        """
        Открывает ВСЕ *.parquet файлы под префиксом как единый dataset.
        Служебные файлы с префиксом '_' или '.' pyarrow.dataset игнорирует.
        Остальные файлы входного префикса должны иметь формат parquet.
        """
        fs = self._filesystem()
        prefix = self._strip_bucket_prefix(s3_prefix)
        return ds.dataset(prefix, filesystem=fs, format="parquet")

    def count_rows(self, s3_prefix: str) -> int:
        """
        Дешёвый подсчёт строк по метаданным всех парт-файлов, без чтения данных.
        """
        dataset = self.open_dataset(s3_prefix)
        return dataset.count_rows()

    def get_schema_columns(self, s3_prefix: str) -> set[str]:
        dataset = self.open_dataset(s3_prefix)
        return set(dataset.schema.names)

    def iter_chunks(self, s3_prefix: str, chunk_size: int):
        """
        Возвращает чанки размером не больше chunk_size строк.
        Границы файлов и row groups могут давать меньшие чанки;
        reader не склеивает остатки разных файлов до полного chunk_size.
        """
        dataset = self.open_dataset(s3_prefix)

        warn_on_oversized_row_groups(dataset, chunk_size, self.logger)

        for batch in dataset.to_batches(batch_size=chunk_size):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*is_sparse is deprecated.*",
                    category=FutureWarning,
                )
                yield batch.to_pandas()

    def get_writer(self, s3_output_prefix: str) -> ParquetPrefixWriter:
        """
        Возвращает писатель, который на каждый write_chunk создаёт новый
        парт-файл под выходным префиксом (по аналогии со Spark парт-файлами),
        а на close() кладёт пустой _SUCCESS маркер.
        """
        return ParquetPrefixWriter(
            fs=self._filesystem(),
            output_prefix=self._strip_bucket_prefix(s3_output_prefix),
        )

    def prefix_has_results(self, s3_prefix: str) -> bool:
        """
        Проверяет наличие parquet или старого маркера _SUCCESS.
        """
        fs = self._filesystem()
        prefix = self._strip_bucket_prefix(s3_prefix)

        try:
            for batch in fs.find(prefix):
                if batch.endswith(".parquet") or batch.rsplit("/", 1)[-1] == _SUCCESS_MARKER:
                    return True
            return False
        except FileNotFoundError:
            # Если Префикса вообще нет
            return False
