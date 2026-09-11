"""S3 с атомарным CAS и управляемыми сбоями для проверки очереди."""

import hashlib
import io
import threading
from dataclasses import replace
from pathlib import Path

import yaml
from botocore.exceptions import ClientError, ReadTimeoutError
from logger_stub import make_logger

from app.core.config import TaskStoreConfig
from app.tasks.state import TaskStore


def config(**overrides):
    raw = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))
    return replace(TaskStoreConfig(**raw["task_store"]), cas_retry_interval_sec=0.0001, **overrides)


class FakeS3:
    def __init__(self):
        self.lock = threading.Lock()
        self.body = None
        self.writes = 0
        self.barrier = None
        self.readers = 0
        self.lose_next_response = False
        self.fail_next = None
        self.requests = []

    @staticmethod
    def error(status):
        return ClientError(
            {
                "Error": {"Code": "NoSuchKey" if status == 404 else str(status)},
                "ResponseMetadata": {"HTTPStatusCode": status},
            },
            "PutObject",
        )

    def get_object(self, **kwargs):
        with self.lock:
            if self.body is None:
                raise self.error(404)
            response = {"Body": io.BytesIO(self.body), "ETag": self.etag}
            barrier = self.barrier
            if barrier is not None:
                self.readers += 1
                if self.readers == 2:
                    self.barrier = None
        if barrier is not None:
            barrier.wait(timeout=5)
        return response

    @property
    def etag(self):
        return '"' + hashlib.md5(self.body).hexdigest() + '"'

    def put_object(self, *, Body, IfMatch=None, IfNoneMatch=None, **kwargs):
        with self.lock:
            self.requests.append((IfMatch, IfNoneMatch))
            if self.fail_next:
                status, self.fail_next = self.fail_next, None
                raise self.error(status)
            if IfNoneMatch == "*" and self.body is not None:
                raise self.error(412)
            if IfMatch is not None and (self.body is None or IfMatch != self.etag.strip('"')):
                raise self.error(412)
            if IfMatch is None and IfNoneMatch != "*":
                raise AssertionError("Безусловный PUT запрещён")
            self.body = Body
            self.writes += 1
            if self.lose_next_response:
                self.lose_next_response = False
                raise ReadTimeoutError(endpoint_url="https://test.invalid")
            return {"ETag": self.etag, "ResponseMetadata": {"HTTPStatusCode": 200}}


def make_store(s3, **overrides):
    return TaskStore(s3, "output", config(**overrides), make_logger())


def enqueue(store, key="request-1", output=None, model="cc"):
    return store.enqueue(key, model, "s3://input/data", output or f"s3://output/{key}", 10)
