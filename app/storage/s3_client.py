"""
Тонкая обёртка над S3 для потокового чтения/записи parquet чанками.

Дизайн для чанковой обработки 2-3 млн строк:
  - open_parquet_file: открывает pyarrow.parquet.ParquetFile через S3FileSystem
    БЕЗ загрузки данных в память — нужно для дешёвой валидации схемы и total_rows.
  - iter_chunks: генератор, отдающий df_chunk по N строк.
  - ParquetChunkWriter: инкрементальный писатель, пишет чанки в output по мере
    готовности, не накапливая весь результат в памяти.

bucket_in/bucket_out из S3Config используются только как дефолтные бакеты для
валидации/документации; сами пути (s3_input_path/s3_output_path) приходят
полностью в запросе /infer в виде "s3://bucket/key".
"""

from dataclasses import dataclass

import pandas as pd
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from app.core.config import S3Config


@dataclass
class S3Client:
    settings: S3Config

    def _filesystem(self) -> pafs.S3FileSystem:
        return pafs.S3FileSystem(
            endpoint_override=self.settings.endpoint_url,
            access_key=self.settings.access_key,
            secret_key=self.settings.secret_key,
            region=self.settings.region,
            scheme="https" if self.settings.use_ssl else "http",
        )

    @staticmethod
    def _strip_bucket_prefix(s3_path: str) -> str:
        # pyarrow.fs.S3FileSystem ожидает путь без "s3://"
        return s3_path.replace("s3://", "", 1)

    def open_parquet_file(self, s3_path: str) -> pq.ParquetFile:
        fs = self._filesystem()
        path = self._strip_bucket_prefix(s3_path)
        return pq.ParquetFile(fs.open_input_file(path))

    def iter_chunks(self, s3_path: str, chunk_size: int):
        """Генератор df_chunk по chunk_size строк, используя iter_batches pyarrow."""
        parquet_file = self.open_parquet_file(s3_path)
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            yield batch.to_pandas()

    def get_writer(self, s3_path: str, schema: pa.Schema) -> "ParquetChunkWriter":
        fs = self._filesystem()
        path = self._strip_bucket_prefix(s3_path)
        sink = fs.open_output_stream(path)
        writer = pq.ParquetWriter(sink, schema)
        return ParquetChunkWriter(writer=writer, sink=sink)


@dataclass
class ParquetChunkWriter:
    writer: pq.ParquetWriter
    sink: object

    def write_chunk(self, df_chunk: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
        self.writer.write_table(table)

    def close(self) -> None:
        self.writer.close()
        self.sink.close()
