"""Проверки parquet и _SUCCESS на файловой системе в памяти, без доступа к S3."""

import unittest
import uuid
from unittest.mock import Mock, patch

import fsspec
import pandas as pd
from logger_stub import make_logger

from app.storage.s3_client import ParquetPrefixWriter, S3Client


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.fs = fsspec.filesystem("memory")
        self.prefix = f"bucket/{uuid.uuid4().hex}"
        self.logger = make_logger()
        self.client = S3Client(Mock(), self.logger)
        patcher = patch.object(S3Client, "_filesystem", return_value=self.fs)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_round_trip_preserves_identifiers_and_success_marker(self):
        writer = ParquetPrefixWriter(self.fs, self.prefix)
        frame = pd.DataFrame(
            {
                "customer_mdm_id": [1, 2],
                "partition_report_dt": ["2026-09-10", "2026-09-10"],
                "prediction": [0.1, 0.2],
            }
        )
        writer.write_chunk(frame)
        self.assertFalse(self.fs.exists(f"{self.prefix}/_SUCCESS"))
        writer.close()
        self.assertTrue(self.fs.exists(f"{self.prefix}/_SUCCESS"))
        # MemoryFileSystem возвращает абсолютные пути, в отличие от S3FileSystem.
        chunks = list(self.client.iter_chunks(f"/{self.prefix}", 1))
        pd.testing.assert_frame_equal(pd.concat(chunks, ignore_index=True), frame)
        self.logger.warn.assert_called_once()

    def test_success_marker_without_parquet_still_blocks_output(self):
        self.fs.pipe(f"{self.prefix}/_SUCCESS", b"")
        self.assertTrue(self.client.prefix_has_results(f"s3://{self.prefix}"))

    def test_empty_output_is_accepted(self):
        self.assertFalse(self.client.prefix_has_results(f"s3://{self.prefix}"))
