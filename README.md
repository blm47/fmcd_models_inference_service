# FMCD Inference Service

FastAPI-сервис инференса FMCD-моделей с фоновой обработкой, S3 I/O чанками и GPU (H100).

## Структура
- `app/main.py` — точка входа, lifespan-загрузка всех моделей на GPU
- `app/api/` — роуты `/infer`, `/tasks/{id}/status`, `/tasks/{id}/abort`
- `app/models/` — ModelBundle, загрузка артефактов (loader.py), батчевый инференс, валидация фич
- `app/tasks/` — состояние задач (in-memory), cancellation, TaskManager
- `app/storage/` — потоковый S3 клиент (чтение/запись parquet чанками)
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
