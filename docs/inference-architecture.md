# Архитектура инференса и интеграция с Airflow

Схемы соответствуют текущей реализации сервиса. DAG, перенос Hadoop ↔ S3 и кастомный оператор показаны как целевая интеграция: их код ещё предстоит реализовать. Модели B и N — примеры масштабирования; в текущем YAML настроена credit_cards. Подразумевается, что каждый под загрузил настроенные модели и имеет один GPU worker.

Диаграммы хранятся как код Mermaid в соседней папке `diagrams/`. Диаграммы последовательностей и состояний показывают UML-взаимодействия; первая схема — архитектурная. Изменения исходников можно сравнивать и ревьюить через Git. Каждый `.mmd` — самостоятельный исходник для Mermaid-рендерера; данный Markdown объединяет их для просмотра.

## 1. Архитектура: DAG, Kubernetes и S3

Исходник: [01-architecture.mmd](diagrams/01-architecture.mmd).

```mermaid
flowchart TB
    H[("Hadoop / HDFS")]

    subgraph AF["Airflow — целевой процесс"]
        direction TB
        subgraph D1["DAG модели credit_cards"]
            A1["Hadoop → S3"] --> B1["Кастомный оператор<br/>POST /infer → ожидание статуса"] --> C1["S3 → Hadoop"]
        end
        subgraph D2["DAG модели B"]
            A2["Hadoop → S3"] --> B2["Кастомный оператор<br/>POST /infer → ожидание статуса"] --> C2["S3 → Hadoop"]
        end
        subgraph DN["DAG модели N"]
            AN["Hadoop → S3"] --> BN["Кастомный оператор<br/>POST /infer → ожидание статуса"] --> CN["S3 → Hadoop"]
        end
    end

    subgraph S3["S3 — общие данные и состояние"]
        IN[("Входные префиксы моделей<br/>part-*.parquet + _SUCCESS")]
        Q[("Один системный JSON<br/>revision + tasks<br/>QUEUED / RUNNING / итог")]
        OUT[("Выходные префиксы моделей<br/>part-*.parquet + _SUCCESS")]
    end

    subgraph K8S["Kubernetes — Deployment"]
        LB["Балансировщик / Service"]
        subgraph P1["ПОД 1 — один процесс Uvicorn"]
            API["FastAPI<br/>приём, статус, отмена"]
            STORE["TaskStore<br/>GET + PUT If-Match"]
            LOOP["Consumer<br/>одна задача одновременно"]
            GPU["GPU worker<br/>чанки → батчи → результат"]
            HB["Heartbeat-поток<br/>только во время выполнения"]
            MON["Монитор<br/>таймауты + retention"]
            LOG["Общий dadm logger<br/>info / warn / error"]
            API --> STORE
            LOOP --> STORE
            LOOP --> GPU
            GPU --> STORE
            HB --> STORE
            MON --> STORE
            GPU -.-> LOG
            API -.-> LOG
        end
        P2["ПОД 2<br/>такие же API, consumer,<br/>GPU, heartbeat и монитор"]
        PN["ПОД N<br/>такие же компоненты"]
        LB --> API
        LB --> P2
        LB --> PN
    end

    H --> A1 & A2 & AN
    A1 & A2 & AN --> IN
    B1 & B2 & BN -->|"HTTP: запуск и polling"| LB
    STORE <-->|"CAS: без lock-файла и DELETE"| Q
    P2 & PN <-->|"Общая очередь через CAS"| Q
    IN -->|"Чтение чанками"| GPU
    GPU -->|"Запись результатов"| OUT
    P2 & PN <-->|"Чтение входа"| IN
    P2 & PN -->|"Запись результата"| OUT
    OUT --> C1 & C2 & CN
    C1 & C2 & CN --> H
```

## 2. Запуск каждого пода

Исходник: [02-startup.mmd](diagrams/02-startup.mmd).

```mermaid
sequenceDiagram
    autonumber
    participant K as Kubernetes
    participant P as Каждый под
    participant Q as S3 — task_state.json
    participant G as GPU
    K->>P: Запуск контейнера
    P->>P: Получить общий dadm logger
    P->>P: YAML — настройки сервиса; ENV — подключение S3
    P->>Q: GET системного JSON
    alt JSON существует
        Q-->>P: Содержимое + ETag
        P->>P: Проверить структуру, сохранить существующие задачи
    else Объект отсутствует
        P->>Q: PUT пустого JSON, If-None-Match = *
        alt Этот под создал объект
            Q-->>P: 200 OK
        else Другой под успел первым
            Q-->>P: 412 Precondition Failed
            P->>Q: GET существующего JSON
            Q-->>P: Очередь другого пода
        end
    end
    Note over P,Q: Ошибка прав или повреждённый JSON — ошибка старта.<br/>Безусловного PUT для восстановления нет.
    P->>G: Загрузить модели и артефакты на device из YAML
    P->>P: Зарегистрировать SIGTERM / SIGINT
    P->>P: Запустить consumer и отдельный монитор
    P-->>K: Readiness OK
    Note over P,G: Один consumer и не более одного GPU-расчёта на под.<br/>Все API доступны и во время инференса.
```

## 3. Успешный расчёт и конкурентный захват

Исходник: [03-inference.mmd](diagrams/03-inference.mmd).

```mermaid
sequenceDiagram
    autonumber
    participant H as Hadoop / HDFS
    participant A as Airflow — DAG модели
    participant S as S3 — parquet
    participant L as Балансировщик
    participant API as API любого пода
    participant Q as S3 — общий JSON очереди
    participant P1 as Consumer ПОД1
    participant P2 as Consumer ПОД2
    participant W as GPU worker победившего пода
    participant M as Heartbeat и монитор

    Note over H,M: Разные DAG выполняют этот сценарий независимо, с разными model_name и префиксами.
    A->>H: Прочитать исходные данные
    H-->>A: Данные модели
    A->>S: Сохранить входные parquet и _SUCCESS
    A->>L: POST /infer: model_name, пути, idempotency_key
    L->>API: Передать запрос любому поду, включая занятый
    API->>Q: Найти прежнюю задачу по ключу
    alt Такой запрос уже принят
        Q-->>API: Существующий task_id и текущий статус
        API-->>A: 202: прежняя задача, без повторного инференса
    else Новый запрос
        API->>API: Проверить пути и наличие модели
        API->>S: Проверить колонки, число строк и пустой выход
        API->>Q: GET JSON + ETag
        API->>API: Проверить ключ, резервирование выхода и лимит очереди
        API->>Q: PUT If-Match: новая задача QUEUED, revision + 1
        Note over API,Q: При CAS-конфликте — перечитать JSON и заново применить изменение.<br/>После потери ответа PUT — проверить, не сохранён ли уже запрос.
        Q-->>API: Запись подтверждена
        API-->>A: 202: task_id, QUEUED, pod_id = null
    end

    par Свободный ПОД1 опрашивает очередь
        P1->>Q: GET: выбрать старейшую доступную QUEUED
        Q-->>P1: Задача T и ETag X
    and Свободный ПОД2 опрашивает очередь
        P2->>Q: GET: выбрать старейшую доступную QUEUED
        Q-->>P2: Та же задача T и ETag X
    end
    P1->>Q: PUT If-Match X: T → RUNNING, pod_id, owner_id
    Q-->>P1: 200 — захват подтверждён
    P2->>Q: PUT If-Match X: попытка забрать T
    Q-->>P2: 412 — версия уже изменилась
    P2->>Q: Перечитать очередь и выбрать другую задачу
    Note over P1,P2: ПОД1 — победитель только в этом примере.<br/>Занятый consumer не забирает следующую задачу.
    P1->>W: Запустить задачу T
    W->>S: Повторно проверить пустой выходной префикс

    par Выполнение инференса
        loop Для каждого входного чанка
            W->>S: Прочитать чанк parquet
            W->>Q: Проверить статус, владельца, таймаут и отмену
            loop Для каждого GPU-батча внутри чанка
                W->>W: Проверить shutdown / сигнал потери задачи
                W->>W: Препроцессинг → GPU forward → калибрация
                W->>W: Проверить shutdown после батча
            end
            W->>Q: Проверить задачу перед записью
            W->>S: Записать part-N.parquet
            opt Настало время обновления прогресса
                W->>Q: CAS: processed_rows, время инференса, updated_at
            end
        end
    and Обслуживание очереди
        loop Heartbeat каждые 30 секунд, пока worker работает
            M->>Q: CAS: обновить heartbeat активной задачи
        end
        Note over M,Q: Монитор каждого пода проверяет таймауты каждые 3 секунды.<br/>Просроченные активные задачи → FAILED.<br/>Retention очищает старые завершённые записи из того же JSON.
    and Кастомный оператор ждёт результат
        loop До финального статуса
            A->>L: GET /tasks/T/status
            L->>API: Запрос может попасть на другой под
            API->>Q: GET общего JSON
            Q-->>API: Статус, прогресс, ETA, error
            API-->>A: Текущее состояние задачи
        end
    end

    W->>Q: CAS: последняя проверка отмены → FINALIZING
    Note over W,Q: Если отмена уже принята — ABORTED, _SUCCESS не публикуется.<br/>Если задача просрочена — FAILED, завершение запрещено.
    W->>S: Сохранить _SUCCESS
    W->>Q: CAS: FINALIZING → DONE
    Note over W,Q: При временной ошибке S3 повторяется сохранение статуса.<br/>Инференс повторно не запускается; существующий FAILED не заменяется.
    A->>L: GET /tasks/T/status
    L->>API: Прочитать итог
    API->>Q: GET
    Q-->>API: DONE
    API-->>A: DONE
    A->>S: Следующая таска читает выходные parquet
    S-->>A: Рассчитанные данные
    A->>H: Записать результат обратно в Hadoop
```

## 4. Ошибки, shutdown, отмена и повтор

Исходник: [04-failures.mmd](diagrams/04-failures.mmd).

```mermaid
sequenceDiagram
    autonumber
    participant K as Kubernetes
    participant W as GPU worker
    participant M as Монитор любого работающего пода
    participant Q as S3 — общий JSON
    participant S as S3 — выходной префикс
    participant A as Airflow — оператор ожидания

    alt Исключение при чтении, расчёте или записи
        W->>W: Зафиксировать ошибку через logger.error
        W->>Q: CAS: FAILED + причина ошибки
    else SIGTERM / SIGINT
        K->>W: Установить сигнал shutdown
        W->>W: Завершить текущую операцию
        W->>W: На границе GPU-батча / чанка проверить сигнал
        W->>Q: CAS: FAILED, error = pod_shutdown
        Note over K,W: Под не начинает новые расчёты.<br/>Grace period в Helm — 120 секунд.<br/>При SIGKILL или OOM итог фиксирует монитор.
    else Авария или зависание
        loop Монитор проверяет JSON каждые 3 секунды
            M->>Q: GET состояния
        end
        alt Нет heartbeat 180 секунд
            M->>Q: CAS: FAILED, error = heartbeat_timeout
        else Нет прогресса 1800 секунд, heartbeat ещё жив
            M->>Q: CAS: FAILED, error = progress_timeout
        end
        Note over M,Q: Проверяются RUNNING, ABORTING и FINALIZING.<br/>QUEUED и финальные статусы не изменяются.<br/>Если нет живых подов или доступа к S3, запись откладывается до восстановления.
    else Отмена через API
        A->>Q: Через API: POST /tasks/T/abort
        alt Задача ещё QUEUED
            Q->>Q: CAS: ABORTED
        else Задача RUNNING
            Q->>Q: CAS: ABORTING + cancel_requested
            W->>Q: Прочитать запрос отмены
            W->>Q: CAS: ABORTED после текущей операции
        else Уже FINALIZING
            Q-->>A: Через API: 409, отмена недоступна
        end
    end

    A->>Q: Через API: прочитать финальный статус
    Q-->>A: FAILED или ABORTED
    A->>S: Очистить выходной префикс и _SUCCESS
    Note over A,S: Системный JSON очереди не удалять.<br/>Политика очистки и повтора реализуется на стороне Airflow.
    A->>Q: Через POST /infer: новый ключ, прежние физические пути
    Q->>Q: CAS: новая задача с новым task_id, QUEUED
    Note over W,S: Ограничение: уже отправленный PUT старого worker может завершиться поздно.<br/>CAS JSON не блокирует запись parquet. При одном пути и только S3<br/>строгая изоляция попыток не гарантируется.
```

## 5. Жизненный цикл одной задачи

Исходник: [05-task-states.mmd](diagrams/05-task-states.mmd).

```mermaid
stateDiagram-v2
    [*] --> QUEUED: Заявка сохранена через CAS
    QUEUED --> RUNNING: Свободный под захватил задачу
    QUEUED --> ABORTED: Отмена до начала
    RUNNING --> ABORTING: Принят запрос отмены
    ABORTING --> ABORTED: Worker завершил текущую операцию
    RUNNING --> FINALIZING: Последний чанк готов, отмены нет
    FINALIZING --> DONE: _SUCCESS записан, итог подтверждён CAS
    RUNNING --> FAILED: Ошибка / shutdown / таймаут
    ABORTING --> FAILED: Ошибка / shutdown / таймаут
    FINALIZING --> FAILED: Ошибка публикации / shutdown до публикации / таймаут
    DONE --> [*]
    FAILED --> [*]
    ABORTED --> [*]
    note right of FAILED
        Финальный статус не возобновляется.
        Airflow создаёт новую задачу
        с новым idempotency_key.
    end note
```

## Контракт и ограничения

- Ответ 202 означает сохранение заявки, а не завершение расчёта. Повтор того же HTTP-запроса использует прежний ключ; новый расчёт после FAILED использует новый ключ.
- Ошибки валидации дают 404/422, конфликт ключа, выходного префикса или лимита очереди — 409, ошибка доступа к хранилищу — 503. Не каждый HTTP 409 означает CAS-конфликт: CAS-конфликты сервис обрабатывает внутри.
- JSON очереди обновляется через GET содержимого и ETag одного объекта → изменение → PUT If-Match. В используемом S3 ETag передаётся без кавычек. При отсутствии изменений PUT не выполняется.
- Значения на схемах взяты из configs/models.yaml: polling 3 с, heartbeat 30 с, heartbeat timeout 180 с, progress timeout 1800 с, обновление прогресса не чаще 10 с, cleanup 3600 с, retention 6 месяцев по 30 суток. Монитор и heartbeat используют разные потоки; lane объединена только для компактности диаграммы.
- Успешный sequence предполагает отсутствие ошибки, отмены и таймаута до фиксации DONE. Ошибочные альтернативы подробно вынесены в отдельную схему.
- После подтверждённой публикации _SUCCESS worker сохраняет DONE; если монитор уже записал FAILED, вернуть DONE нельзя. S3-результат и JSON не составляют общей транзакции.
- Одинаковый физический выходной путь сохраняется по принятому контракту. Очистка и повтор после таймаута не имеют строгой защиты от позднего PUT старого пода. Сервис проверяет статус перед записью и сигнал остановки между GPU-батчами, но не отменяет запрос, уже принятый S3.
- Перенос из S3 обратно в Hadoop запускается только после DONE. Финальные FAILED/ABORTED сами по себе не возвращают задачу в очередь: повтор инициирует Airflow.
