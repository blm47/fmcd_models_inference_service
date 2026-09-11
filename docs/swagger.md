# FMCD Inference Service

Снимок API из FastAPI. Обновление: `python scripts/export_openapi.py`.

Раскройте метод или DTO для просмотра. Примеры запросов нужно адаптировать
к своим бакетам и модели. Отправка запросов доступна в `/docs` сервиса.

Асинхронный GPU-инференс над parquet в S3. Любой под принимает заявки, свободный worker забирает их из общей очереди S3.

**Airflow:** подготовить вход → `POST /infer` → опрашивать `GET /tasks/{task_id}/status` → при `DONE` забрать результат в Hadoop.

**Статусы:** `QUEUED → RUNNING → FINALIZING → DONE`. Ошибка или таймаут переводит задачу в `FAILED`. Отмена: `QUEUED → ABORTED` или `RUNNING → ABORTING → ABORTED`.

Повтор HTTP-запроса использует прежний ключ идемпотентности. Новая попытка расчёта требует нового ключа и очищенного выходного префикса. Не удаляйте системный файл очереди. `FAILED` по таймауту не подтверждает остановку старого PUT: повтор в тот же физический путь не имеет строгой изоляции.

## Методы

<details>
<summary>POST /infer — Поставить расчёт в очередь</summary>

Принимает заявку независимо от занятости GPU текущего пода.

До постановки проверяет модель, входные parquet и выходной префикс.
Входные данные должны оставаться неизменными до завершения расчёта.
Выход не должен содержать parquet или `_SUCCESS` и пересекаться с входом,
системным файлом либо выходом незавершённой задачи.

Одинаковый `idempotency_key` и параметры возвращают исходную задачу,
в том числе завершённую, пока она хранится в истории. Ответ 202 означает
принятие заявки, а не завершение расчёта. Новая задача имеет статус QUEUED.

**Parameters**

Нет параметров пути или query.

**Request body**

`application/json` · [InferRequest](#inferrequest) · обязательное тело

Пример запроса:

```json
{
  "idempotency_key": "credit_cards/run-1/infer/attempt-1",
  "model_name": "credit_cards",
  "s3_input_path": "s3://input-bucket/data/run-1",
  "s3_output_path": "s3://output-bucket/results/run-1"
}
```


**Responses**

| HTTP | Описание | Тело ответа |
| --- | --- | --- |
| 202 | Заявка сохранена или найдена ранее принятая попытка | application/json: [TaskAcceptedResponse](#taskacceptedresponse) |
| 404 | Модель или входной префикс не найдены | application/json: [ErrorResponse](#errorresponse) |
| 409 | Конфликт ключа, занятый выходной префикс или полная очередь | application/json: [ErrorResponse](#errorresponse) |
| 422 | Некорректные параметры, колонки или непустой выход | application/json: [ErrorResponse](#errorresponse) |
| 503 | S3 недоступен. Повторите запрос с прежним ключом | application/json: [ErrorResponse](#errorresponse) |

</details>

<details>
<summary>GET /tasks/{task_id}/status — Получить статус и прогресс расчёта</summary>

Читает общую очередь, поэтому запрос можно направить на любой под.

Airflow опрашивает этот метод до DONE, FAILED или ABORTED.
Результат готов к чтению только при DONE. ETA является оценкой, а не дедлайном.
`executor_alive` вычисляется по heartbeat и не подтверждает физическую
остановку процесса при значении false.

**Parameters**

| Имя | Где | Тип | Обязательно | Описание |
| --- | --- | --- | --- | --- |
| task_id | path | string | да | — |

**Responses**

| HTTP | Описание | Тело ответа |
| --- | --- | --- |
| 200 | Successful Response | application/json: [TaskStatusResponse](#taskstatusresponse) |
| 503 | Не удалось прочитать или обновить очередь S3 | application/json: [ErrorResponse](#errorresponse) |
| 404 | Задача не найдена или удалена из истории | application/json: [ErrorResponse](#errorresponse) |
| 422 | Validation Error | application/json: [HTTPValidationError](#httpvalidationerror) |

</details>

<details>
<summary>POST /tasks/{task_id}/abort — Запросить отмену расчёта</summary>

QUEUED отменяется сразу, RUNNING переходит в ABORTING.

Worker завершает текущую операцию и подтверждает ABORTED при проверке отмены.
Ответ ABORTING ещё не означает остановку. Для завершённой задачи возвращается
её текущий статус. Этот метод не удаляет выходные файлы и не запускает повтор.

**Parameters**

| Имя | Где | Тип | Обязательно | Описание |
| --- | --- | --- | --- | --- |
| task_id | path | string | да | — |

**Responses**

| HTTP | Описание | Тело ответа |
| --- | --- | --- |
| 200 | Successful Response | application/json: [TaskAbortResponse](#taskabortresponse) |
| 503 | Не удалось прочитать или обновить очередь S3 | application/json: [ErrorResponse](#errorresponse) |
| 404 | Задача не найдена | application/json: [ErrorResponse](#errorresponse) |
| 409 | Задача FINALIZING уже публикует результат | application/json: [ErrorResponse](#errorresponse) |
| 422 | Validation Error | application/json: [HTTPValidationError](#httpvalidationerror) |

</details>

<details>
<summary>GET /tasks/active — Получить все незавершённые задачи</summary>

Все незавершённые задачи, включая очередь и потерявших heartbeat исполнителей.

**Parameters**

Нет параметров пути или query.

**Responses**

| HTTP | Описание | Тело ответа |
| --- | --- | --- |
| 200 | Successful Response | application/json: [ActiveTasksResponse](#activetasksresponse) |
| 503 | Не удалось прочитать или обновить очередь S3 | application/json: [ErrorResponse](#errorresponse) |

</details>

<details>
<summary>GET /healthz/liveness — Проверить работу потоков пода</summary>

200 при работающих consumer и monitor, иначе 503. Доступ к S3 не проверяет.

**Parameters**

Нет параметров пути или query.

**Responses**

| HTTP | Описание | Тело ответа |
| --- | --- | --- |
| 200 | Successful Response | application/json: [HealthResponse](#healthresponse) |
| 503 | unavailable | application/json: [HealthResponse](#healthresponse) |

</details>

<details>
<summary>GET /healthz/readiness — Проверить готовность пода</summary>

200 при работающих consumer и monitor, иначе 503. Доступ к S3 не проверяет.

**Parameters**

Нет параметров пути или query.

**Responses**

| HTTP | Описание | Тело ответа |
| --- | --- | --- |
| 200 | Successful Response | application/json: [HealthResponse](#healthresponse) |
| 503 | unavailable | application/json: [HealthResponse](#healthresponse) |

</details>

<details>
<summary>GET /health — Проверить состояние сервиса</summary>

200 при работающих consumer и monitor, иначе 503. Доступ к S3 не проверяет.

**Parameters**

Нет параметров пути или query.

**Responses**

| HTTP | Описание | Тело ответа |
| --- | --- | --- |
| 200 | Successful Response | application/json: [HealthResponse](#healthresponse) |
| 503 | unavailable | application/json: [HealthResponse](#healthresponse) |

</details>

## Schemas

Обязательное поле должно присутствовать в JSON. Тип `null` отдельно
указывает, что значение может быть пустым.

### ActiveTaskSummary

<details>
<summary>ActiveTaskSummary — раскрыть схему</summary>

| Поле | Тип | Обязательно | Описание | Ограничения и примеры |
| --- | --- | --- | --- | --- |
| task_id | string | да | Идентификатор задачи | — |
| pod_id | string &#124; null | да | Под-исполнитель, null до назначения | — |
| model_name | string | да | Имя модели | — |
| status | [TaskStatus](#taskstatus) | да | Один из незавершённых статусов | — |
| progress_pct | number | да | Сохранённый прогресс в процентах | — |
| eta_seconds | number &#124; null | да | Оценка оставшихся секунд, если доступна | — |
| executor_alive | boolean | да | Признак актуального heartbeat исполнителя | — |

</details>

### ActiveTasksResponse

<details>
<summary>ActiveTasksResponse — раскрыть схему</summary>

| Поле | Тип | Обязательно | Описание | Ограничения и примеры |
| --- | --- | --- | --- | --- |
| active_tasks | array&lt;ActiveTaskSummary&gt; | да | Незавершённые задачи всех подов | — |

</details>

### ErrorResponse

<details>
<summary>ErrorResponse — раскрыть схему</summary>

| Поле | Тип | Обязательно | Описание | Ограничения и примеры |
| --- | --- | --- | --- | --- |
| detail | string &#124; object &#124; array&lt;any&gt; | да | Сообщение ошибки, отсутствующие колонки или список ошибок валидации | — |

</details>

### HTTPValidationError

<details>
<summary>HTTPValidationError — раскрыть схему</summary>

| Поле | Тип | Обязательно | Описание | Ограничения и примеры |
| --- | --- | --- | --- | --- |
| detail | array&lt;ValidationError&gt; | нет | — | — |

</details>

### HealthResponse

<details>
<summary>HealthResponse — раскрыть схему</summary>

| Поле | Тип | Обязательно | Описание | Ограничения и примеры |
| --- | --- | --- | --- | --- |
| status | string | да | ok или unavailable | examples: [&quot;ok&quot;] |

</details>

### InferRequest

<details>
<summary>InferRequest — раскрыть схему</summary>

| Поле | Тип | Обязательно | Описание | Ограничения и примеры |
| --- | --- | --- | --- | --- |
| idempotency_key | string | да | Стабильный ключ одной попытки расчёта Airflow | maxLength: 256; minLength: 1; examples: [&quot;credit_cards/run-1/infer/attempt-1&quot;] |
| model_name | string | да | Имя модели из configs/models.yaml | minLength: 1; examples: [&quot;credit_cards&quot;] |
| s3_input_path | string | да | Входной префикс внутри S3_BUCKET_IN с подготовленными parquet | examples: [&quot;s3://input-bucket/data/run-1&quot;] |
| s3_output_path | string | да | Выходной префикс внутри S3_BUCKET_OUT без parquet и _SUCCESS | examples: [&quot;s3://output-bucket/results/run-1&quot;] |

</details>

### TaskAbortResponse

<details>
<summary>TaskAbortResponse — раскрыть схему</summary>

| Поле | Тип | Обязательно | Описание | Ограничения и примеры |
| --- | --- | --- | --- | --- |
| task_id | string | да | Идентификатор задачи | — |
| status | [TaskStatus](#taskstatus) | да | Статус после запроса отмены. ABORTING требует ожидания | — |

</details>

### TaskAcceptedResponse

<details>
<summary>TaskAcceptedResponse — раскрыть схему</summary>

| Поле | Тип | Обязательно | Описание | Ограничения и примеры |
| --- | --- | --- | --- | --- |
| task_id | string | да | Идентификатор задачи для опроса статуса и отмены | — |
| pod_id | string &#124; null | да | Под-исполнитель, null до назначения | — |
| status | [TaskStatus](#taskstatus) | да | Текущий статус, у новой заявки QUEUED | — |
| total_rows | integer | да | Количество строк во входных данных | — |

</details>

### TaskStatus

<details>
<summary>TaskStatus — раскрыть схему</summary>

Значения: `QUEUED`, `RUNNING`, `DONE`, `FAILED`, `ABORTING`, `ABORTED`, `FINALIZING`


</details>

### TaskStatusResponse

<details>
<summary>TaskStatusResponse — раскрыть схему</summary>

| Поле | Тип | Обязательно | Описание | Ограничения и примеры |
| --- | --- | --- | --- | --- |
| task_id | string | да | Идентификатор задачи | — |
| model_name | string | да | Имя модели из конфигурации | — |
| pod_id | string &#124; null | да | Назначенный под, null до захвата задачи | — |
| status | [TaskStatus](#taskstatus) | да | DONE, FAILED и ABORTED — финальные статусы | — |
| processed_rows | integer | да | Количество обработанных строк по сохранённому прогрессу | — |
| total_rows | integer | да | Количество входных строк | — |
| progress_pct | number | да | Прогресс в процентах. 100 ещё не означает DONE | — |
| eta_seconds | number &#124; null | да | Оценка оставшихся секунд, null если недоступна | — |
| error | string &#124; null | да | Причина ошибки или таймаута, null при отсутствии | — |
| heartbeat_at | number &#124; null | да | Последний heartbeat: Unix timestamp в секундах | — |
| executor_alive | boolean | да | Признак актуального heartbeat, не проверка процесса | — |
| s3_output_path | string | да | Физический S3-префикс результата. Читать после DONE | — |
| created_at | number | да | Создание задачи: Unix timestamp в секундах | — |
| started_at | number &#124; null | да | Начало расчёта: Unix timestamp, null до старта | — |
| finished_at | number &#124; null | да | Завершение: Unix timestamp, null до завершения | — |

</details>

### ValidationError

<details>
<summary>ValidationError — раскрыть схему</summary>

| Поле | Тип | Обязательно | Описание | Ограничения и примеры |
| --- | --- | --- | --- | --- |
| loc | array&lt;string &#124; integer&gt; | да | — | — |
| msg | string | да | — | — |
| type | string | да | — | — |
| input | any | нет | — | — |
| ctx | object | нет | — | — |

</details>
