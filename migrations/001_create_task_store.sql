-- Начальная схема очереди. {schema} и {table} подставляются как SQL identifiers.
-- Миграция выполняется целиком в одной транзакции до старта сервиса.
CREATE SCHEMA IF NOT EXISTS {schema};

CREATE TABLE {table} (
    task_id text PRIMARY KEY,
    model_name text NOT NULL,
    s3_input_path text NOT NULL,
    s3_output_path text NOT NULL,
    status text NOT NULL CHECK (status IN (
        'QUEUED', 'RUNNING', 'FINALIZING', 'DONE', 'FAILED', 'ABORTING', 'ABORTED'
    )),
    total_rows bigint CHECK (total_rows >= 0),
    calc_utilization boolean NOT NULL DEFAULT false,
    idempotency_key text UNIQUE,
    pod_id text,
    processed_rows bigint NOT NULL DEFAULT 0 CHECK (processed_rows >= 0),
    -- Время в Unix seconds, единый формат с API и S3 backend.
    created_at double precision NOT NULL,
    started_at double precision,
    last_modified double precision NOT NULL,
    finished_at double precision,
    error text,
    inference_elapsed_sec double precision NOT NULL DEFAULT 0,
    owner_id text,
    cancel_requested boolean NOT NULL DEFAULT false
);

-- Опрос очереди не сканирует неограниченную историю завершённых задач.
CREATE INDEX ON {table} (created_at, task_id)
    WHERE status NOT IN ('DONE', 'FAILED', 'ABORTED');
CREATE INDEX ON {table} (last_modified)
    WHERE status IN ('RUNNING', 'ABORTING', 'FINALIZING');
CREATE UNIQUE INDEX ON {table} (owner_id)
    WHERE owner_id IS NOT NULL AND status IN ('RUNNING', 'ABORTING', 'FINALIZING');
