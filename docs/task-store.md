# Хранение очереди: S3 или PostgreSQL

## Настройка

В `configs/models.yaml` задаётся `task_store.backend: s3` или `pg`.
Helm получает этот же YAML через `--set-file serviceConfig=configs/models.yaml`.
Все поды одного сервиса используют одинаковый backend и одну очередь.
S3 остаётся хранилищем parquet при любом backend очереди.

```yaml
task_store:
  backend: pg
  task_timeout_sec: 1800
  # Остальные настройки сохраняются из configs/models.yaml.
```

Подключение PostgreSQL задаётся ENV через Helm:

| ENV | Helm values / Secret | Назначение |
| --- | --- | --- |
| POSTGRES_URL | postgres.url | `postgresql://host:5432/database`, без credentials и query |
| POSTGRES_LOGIN | postgres.existingSecretName, ключ login | Пользователь |
| POSTGRES_PASSWORD | postgres.existingSecretName, ключ password | Пароль |
| POSTGRES_SCHEMA | postgres.schema | Схема |
| POSTGRES_TABLE_NAME | postgres.tableName | Таблица |
| POSTGRES_SSLMODE | postgres.sslmode | Режим libpq TLS; Helm: require, без ENV: prefer |

Для `verify-full` доверенный CA настраивается стандартным `PGSSLROOTCERT` и
монтируется в контейнер. Имена схемы и таблицы экранируются как SQL identifiers,
значения передаются параметрами. Пароль не включается в repr конфигурации и ошибки PG.
ENV PostgreSQL не требуется и не читается при выборе `s3`.

## Начальная миграция

DDL: [001_create_task_store.sql](../migrations/001_create_task_store.sql).
Одна строка соответствует одной задаче, поля состояния — отдельные колонки.
Время хранится в Unix seconds, как в JSON API и S3. Есть индексы незавершённых
задач, уникальность ключа идемпотентности и единственного расчёта процесса.

После задания `POSTGRES_SCHEMA` и `POSTGRES_TABLE_NAME` подготовьте SQL:

```sh
python scripts/migrate_task_store.py > task_store.sql
```

Для выполнения задайте все ENV подключения и запустите:

```sh
python scripts/migrate_task_store.py --apply
```

Миграция выполняется в одной транзакции и не удаляет существующую таблицу.
Повторный запуск на существующей таблице завершится ошибкой без частичного изменения.
Автоматически при старте сервиса DDL не выполняется. Миграционная роль должна иметь
права CREATE; сервисной роли достаточно USAGE схемы и SELECT, INSERT, UPDATE таблицы.
Если миграционная и сервисная роли различаются, DBA выдаёт эти права после миграции.

Текущие хранилища создаются заново: миграция старых записей и совместимость со старым
JSON не предусмотрены. Переключение backend не переносит задачи. Перед переключением
остановите старые worker и запустите все поды с новой конфигурацией.

## Контракт backend

`TaskStore` содержит логику очереди: enqueue, claim, прогресс, отмену, завершение,
проверку таймаутов. Он получает `TaskBackend` через конструктор и не работает с
boto3, SQL, ETag или форматом файла.

Контракт включает `get`, `get_by_key`, `list_active`, `mutate`, `initialize`, `close`
и опциональный retention через `supports_retention`/`cleanup`.
`mutate` атомарно применяет callback к снимку всех незавершённых задач и нужной
завершённой записи по ID/ключу. Callback не выполняет I/O и допускает повтор.

- **S3:** один JSON с revision; GET → изменение → условный PUT с If-Match.
  Конфликты и временные ошибки повторяются с перечитыванием состояния.
  Завершённые записи удаляются после `retention_months` (по умолчанию 6).
- **PG:** короткая транзакция READ COMMITTED с блокировкой таблицы
  SHARE ROW EXCLUSIVE до чтения снимка. Это сериализует изменения очереди,
  включая проверку общего лимита и пересечения выходных путей. Обычные SELECT
  продолжают работать. В таблицу записываются только изменённые строки.
  История не загружается целиком при опросе и не удаляется сервисом.

Это простая реализация для небольшой очереди инференса: запись разных задач
сериализуется, но GPU, чтение parquet и публикация результата идут вне транзакции.
Семантика блокировки описана в [документации PostgreSQL](https://www.postgresql.org/docs/current/explicit-locking.html).
Контекст соединения подтверждает транзакцию при успехе и откатывает её при ошибке:
[документация Psycopg](https://www.psycopg.org/psycopg3/docs/basic/transactions.html).

PG открывает отдельное короткое соединение на операцию. Connect ограничен
`connect_timeout_sec`, SQL — `read_timeout_sec`, ожидание блокировки — также
`cas_timeout_sec` (при меньшем statement timeout сработает он).
При неподтверждённой операции PG API возвращает 503. Клиент повторяет запрос с тем
же ключом; worker повторяет работу штатным циклом, проверка очереди повторяется при следующем readiness. Потеря ответа COMMIT
не означает, что запись не произошла: повтор enqueue/claim восстанавливает результат.

## Heartbeat, last_modified и Kubernetes-пробы

Во время выполнения задачи отдельный heartbeat обновляет `last_modified` каждые
`heartbeat_interval_sec` (по умолчанию 30 секунд), независимо от load/infer/I/O.
Он останавливается при выходе worker, в том числе после ошибки или отмены.
Claim, сохранение total_rows, прогресса и финального статуса также обновляют это поле.
Отдельные поля heartbeat_at/updated_at не требуются; формат хранения не меняется.

Если RUNNING, ABORTING или FINALIZING не обновлялась `task_timeout_sec` (1800 секунд),
задача получает FAILED с `error=task_timeout: ...`. Просроченный heartbeat не оживляет
задачу. Опрос статуса и запрос отмены RUNNING-задачи не продлевают её жизнь.
Таймаут должен превышать интервал heartbeat и ожидаемые задержки хранилища.
Он показывает потерю связи с исполнителем: зависший infer при работающем heartbeat
сам по себе не обнаруживается. Таймаут отсутствия прогресса отдельно не проверяется.

Потока `monitor_queue` нет. Проверку запускает Kubernetes через health endpoints:

| Ручка | Проверка | При сбое |
| --- | --- | --- |
| `/healthz/liveness` | Consumer работает, остановка не запрошена; без S3/PG | 503, Kubernetes может перезапустить контейнер |
| `/healthz/readiness` | Consumer работает; проверка таймаутов очереди и периодический retention S3 | 503, под исключается из трафика |
| `/health` | Alias readiness | То же, что readiness |

Readiness обращается к очереди и при занятом GPU. Cleanup S3 запускается не чаще
`cleanup_interval_sec`, PG не очищается. Одновременные readiness не накапливают
обращения к медленному хранилищу: дополнительный вызов получает 503.
После ошибки следующая проба повторяет проверку.

Helm вызывает readiness раз в 15 секунд, liveness — раз в 30 секунд.
`probes.readinessTimeoutSeconds` по умолчанию 120: учитывает проверку и cleanup
с CAS retry. При увеличении storage timeout увеличьте также timeout пробы.
Liveness имеет timeout 5 секунд и не зависит от доступности S3/PG.
Семантика проб: [документация Kubernetes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/).

Вне Kubernetes нужно периодически вызывать readiness; без этих запросов нет
фоновой проверки чужих задач и retention. После остановки всех подов проверка
возобновится с первой успешной readiness. `executor_alive` вычисляется по
`last_modified`, но не доказывает физическую остановку старого процесса.

## Проверки PostgreSQL

Unit-тесты запускаются вместе с общей suite. Для интеграционных тестов задайте
`TEST_POSTGRES_URL`, `TEST_POSTGRES_LOGIN`, `TEST_POSTGRES_PASSWORD` и при необходимости
`TEST_POSTGRES_SSLMODE`, затем:

```sh
python -m unittest discover -s tests -p test_postgres.py
```

Используйте отдельную тестовую БД: каждый тест создаёт случайную схему `fmcd_test_*`
и удаляет только эту схему после завершения. Нужны права CREATE SCHEMA.
Проверяются конкурентные заявки/claim, лимит очереди, конфликты путей, rollback,
идемпотентность завершённой задачи, отсутствие retention и запрет оживления по таймауту.
Без `TEST_POSTGRES_URL` интеграционные тесты явно пропускаются.
