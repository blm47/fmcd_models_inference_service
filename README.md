# FMCD Inference Service

FastAPI-сервис инференса FMCD-моделей с фоновой обработкой, потоковым S3 I/O
по ПРЕФИКСАМ (Spark-style: _SUCCESS + part-*.parquet) и GPU (H100).

## Важно: контракт S3-путей
`s3_input_path` и `s3_output_path` в запросе `/infer` - это ПРЕФИКСЫ (папки),
а не пути к одиночным файлам:
- `s3_input_path` - папка, куда Spark сохранил DataFrame (
  `_SUCCESS` + множество `part-*.parquet`). Сервис читает ВСЕ part-файлы
  под этим префиксом как единый датасет и режет их на чанки по `chunk_size`
  строк (границы чанков не совпадают с границами part-файлов).
- `s3_output_path` - папка, куда сервис пишет результат: по одному
  `part-{idx:05d}.parquet` на каждый chunk_size строк, и `_SUCCESS` маркер
  после успешного завершения (маркер НЕ пишется при ошибке/аборте - это
  сигнал для downstream, что выгрузка неполная).

## Структура
- `app/main.py` — точка входа, lifespan-загрузка всех моделей на GPU
- `app/api/` — роуты `/infer`, `/tasks/{id}/status`, `/tasks/{id}/abort`
- `app/models/` — ModelBundle, загрузка артефактов (loader.py), батчевый инференс, валидация фич
- `app/tasks/` — состояние задач (in-memory), cancellation, TaskManager
- `app/storage/s3_client.py` — потоковое чтение/запись parquet ПРЕФИКСОВ через pyarrow.dataset
- `fmcd/` — существующий пакет пользователя (schema/data/model), копируется как есть
- `artifacts/<model_name>/` — веса и препроцессинг-артефакты каждой модели, вшиты в образ
- `configs/models.yaml` — список моделей + инференс/S3(имена env)/task конфигурация
- `deploy/` — Dockerfile и Helm chart

## Конфигурация через env
Все значения S3 (эндпоинт, ключи, бакеты) читаются из переменных окружения,
имена которых заданы в `configs/models.yaml -> s3.*_env`. Helm прокидывает
сами значения через `env:` в Deployment (ConfigMap для несекретных значений,
Secret для access/secret key).

## Локальный запуск
```bash
pip install -r requirements.txt
export S3_ENDPOINT_URL=... S3_BUCKET_IN=... S3_BUCKET_OUT=... S3_ACCESS_KEY=... S3_SECRET_KEY=...
uvicorn app.main:app --reload
```

## Сборка образа
```bash
docker build -f deploy/Dockerfile -t fmcd-inference-service:0.1.0 .
```

## Деплой через Helm
```bash
helm install fmcd-inference deploy/helm -f deploy/helm/values.yaml
```
