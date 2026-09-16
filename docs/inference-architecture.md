# Работа сервиса инференса: архитектура, последовательность и статусы

Диаграммы предназначены для аналитиков, поддержки и технических специалистов.
Airflow управляет запуском Spark jobs и инференса; сервис выполняет GPU-расчёт,
S3 хранит вход и выход; общая очередь использует S3 или PostgreSQL.

В кастомной библиотеке уже есть операторы hadoop_2_S3 и S3_2_Hadoop:
через spark-submit они запускают Spark jobs, читающие parquet из одного хранилища
и записывающие в другое. Данные не проходят через Airflow workers или XCom.
У каждой модели свой DAG, очередь и поды общие. TaskStore использует backend S3 или PostgreSQL;
Выбор задаётся в YAML: `task_store.backend: s3` или `pg`.

## 1. Общая архитектура

При старте под вызывает заглушку скачивания артефактов; веса загружает worker только после захвата задачи. Затем API принимает заявки, свободный worker забирает расчёты, а readiness-запросы Kubernetes запускают проверку таймаутов. Очередь физически хранится в выбранном backend — на схеме она включена в границу сервиса как его общий компонент.

```mermaid
flowchart TB
    H[("Hadoop")]
    IN[("Входные parquet S3")]
    OUT[("Результаты parquet S3")]
    subgraph AF["Airflow — управление отдельным DAG каждой модели"]
        E["hadoop_2_S3"]
        OP["GPUModelInferenceOperator<br/>Запуск и ожидание расчёта"]
        R["S3_2_Hadoop после DONE"]
        E --> OP --> R
    end
    subgraph SP["Spark — передача данных между хранилищами"]
        JOBIN["Spark job Hadoop → S3"]
        JOBOUT["Spark job S3 → Hadoop"]
    end
    E -.->|"spark-submit / статус"| JOBIN
    R -.->|"spark-submit / статус"| JOBOUT
    H -->|"parquet"| JOBIN --> IN
    OUT -->|"parquet"| JOBOUT --> H
    subgraph K["Kubernetes — сервис инференса"]
        API["API через балансировщик<br/>Заявки, статус, отмена"]
        Q[("TaskStore: S3 / PostgreSQL<br/>Статусы и прогресс")]
        P1["Под 1 — один worker"]
        P2["Под 2 — один worker"]
        M["Kubelet вызывает readiness<br/>Проверка таймаутов в ручке"]
        API <--> Q
        Q -->|"Задачу берёт свободный под"| P1
        Q --> P2
        P1 & P2 -->|"Прогресс и heartbeat"| Q
        M <--> Q
    end
    OP <-->|"HTTP: запуск и статус"| API
    IN --> P1 & P2
    P1 & P2 --> OUT
```

[Исходник диаграммы](diagrams/01-architecture.mmd).

## 2. Успешный расчёт — UML sequence

Показан путь одной новой заявки без ошибок и отмены. Балансировщик может направить каждый запрос на любой под. Ответ 202 подтверждает приём заявки, а DONE — завершение расчёта.

```mermaid
sequenceDiagram
    autonumber
    participant H as Hadoop
    participant A as Airflow — DAG модели
    participant SP as Spark jobs — spark-submit
    participant S as S3 — данные
    participant API as API сервиса
    participant Q as Общая очередь TaskStore
    participant W as Свободный GPU-под

    Note over A,SP: Кастомные операторы hadoop_2_S3 и S3_2_Hadoop запускают Spark jobs.<br/>Airflow передаёт параметры и получает статус, parquet через него не проходит.
    A->>SP: hadoop_2_S3 — spark-submit
    SP->>H: Прочитать исходные parquet / выполнить SQL
    H-->>SP: Входные данные
    SP->>S: Записать подготовленные parquet
    SP-->>A: Статус завершения выгрузки
    A->>API: POST /infer — модель, пути, ключ попытки
    API->>Q: Сохранить новую заявку
    Note over API,Q: Статус QUEUED — ожидает свободного исполнителя
    API-->>A: 202 — заявка принята, task_id

    Note over Q,W: Под принимает HTTP-запросы даже при занятом GPU.<br/>Один worker на поде, одна задача за раз.
    W->>Q: Забрать старейшую доступную заявку
    Q-->>W: Заявка закреплена за этим подом — RUNNING
    W->>W: При calc_utilization=true запустить sampler в памяти
    W->>W: Создать ModelRunner и загрузить модель с артефактами
    W->>S: Проверить входные колонки и количество строк
    W->>Q: Сохранить total_rows
    W->>S: Проверить пустой выходной префикс

    par Обработка данных
        loop Для каждого чанка
            W->>S: Прочитать часть входных parquet
            W->>W: Подготовить данные и выполнить infer-батчи
            W->>Q: Проверить, что задача активна и не отменена
            W->>S: Записать часть результата модели
            W->>Q: Периодически сохранить прогресс
        end
    and Heartbeat исполнителя
        loop Пока worker выполняет задачу
            W->>Q: Обновить last_modified
        end
    and Сбор статистики на поде
        loop При calc_utilization=true каждую секунду до close
            W->>W: CPU, RAM, GPU, GPU_RAM — min/max/mean/median
        end
    and Ожидание в Airflow
        loop До финального статуса
            A->>API: GET /tasks/task_id/status
            API->>Q: Прочитать состояние
            Q-->>API: Статус, прогресс, last_modified, причина ошибки
            API-->>A: Текущее состояние задачи
        end
    end

    W->>Q: Проверить отмену и начать завершение — FINALIZING
    W->>S: Записать маркер _SUCCESS
    W->>Q: Подтвердить DONE и last_modified
    W->>W: close в finally, остановить сбор утилизации
    W->>W: При включённом sampler вывести итог в logger
    A->>API: Проверить итоговый статус
    API->>Q: Прочитать состояние
    Q-->>API: DONE
    API-->>A: DONE
    A->>SP: S3_2_Hadoop — spark-submit
    SP->>S: Прочитать рассчитанные parquet
    S-->>SP: Результаты модели
    SP->>H: Записать parquet результата в Hadoop
    SP-->>A: Статус завершения переноса
```

[Исходник диаграммы](diagrams/03-inference.mmd).

## 3. Ошибка, отмена и повтор — UML sequence

Показаны альтернативы для выполняющейся задачи. Отмена ожидающей заявки сразу даёт ABORTED. После перехода в FINALIZING запрос отмены отклоняется — публикация уже началась.

```mermaid
sequenceDiagram
    autonumber
    participant A as Airflow
    participant API as API сервиса
    participant Q as Общая очередь TaskStore
    participant W as GPU-под
    participant M as Readiness по запросу Kubelet
    participant S as S3 — выходные данные

    Note over Q,W: Задача выполняется: RUNNING
    alt Ошибка расчёта или остановка пода
        W->>W: Завершить текущую операцию и остановить расчёт
        W->>Q: FAILED — сохранить причину ошибки
    else Исполнитель перестал обновлять last_modified
        M->>Q: Проверить last_modified
        M->>Q: При превышении таймаута сохранить FAILED
    else Запрошена отмена
        A->>API: POST /tasks/task_id/abort
        API->>Q: ABORTING — отмена принята
        W->>Q: Прочитать запрос отмены
        W->>W: Завершить текущую операцию
        W->>Q: ABORTED — расчёт остановлен
    end

    A->>API: Проверить статус
    API->>Q: Прочитать итог
    Q-->>API: FAILED или ABORTED и причина
    API-->>A: Расчёт не завершён успешно
    A->>S: Очистить выходной префикс этой попытки
    A->>API: Новый запрос с новым ключом попытки
    API->>Q: Новая задача — QUEUED
    Note over A,Q: Новая попытка начинается с начала.<br/>Старая задача сохраняет финальный статус.
```

[Исходник диаграммы](diagrams/04-failures.mmd).

## 4. Переходы статусов — UML state

Это жизненный цикл одной задачи. Повтор Airflow создаёт другой task_id и начинает новый жизненный цикл с QUEUED.

```mermaid
stateDiagram-v2
    state "QUEUED: ожидает исполнителя" as QUEUED
    state "RUNNING: выполняется расчёт" as RUNNING
    state "FINALIZING: публикуется результат" as FINALIZING
    state "DONE: успешно завершена" as DONE
    state "ABORTING: запрошена отмена" as ABORTING
    state "ABORTED: отменена" as ABORTED
    state "FAILED: завершена с ошибкой" as FAILED

    [*] --> QUEUED: Заявка сохранена
    QUEUED --> RUNNING: Свободный под забрал задачу
    RUNNING --> FINALIZING: Все чанки обработаны, отмены нет
    FINALIZING --> DONE: _SUCCESS и итоговый статус сохранены
    QUEUED --> ABORTED: Отмена до запуска
    RUNNING --> ABORTING: Принят запрос отмены
    ABORTING --> ABORTED: Исполнитель остановил расчёт
    RUNNING --> FAILED: Ошибка, shutdown или таймаут
    ABORTING --> FAILED: Ошибка, shutdown или таймаут
    FINALIZING --> FAILED: Ошибка, таймаут или shutdown до публикации
    DONE --> [*]
    ABORTED --> [*]
    FAILED --> [*]

    note right of FAILED
        Повтор создаёт новую задачу.
        Перехода FAILED → RUNNING нет.
    end note
```

[Исходник диаграммы](diagrams/05-task-states.mmd).

## Правила, важные при сопровождении

- **Очередь общая.** S3 согласует изменения через CAS одного JSON, PostgreSQL — транзакцией с блокировкой таблицы. Занятость пода не мешает ему принять новую заявку.
- **Повтор запроса и повтор расчёта различаются.** После потери HTTP-ответа Airflow использует прежний ключ и получает прежнюю задачу. После FAILED или ABORTED новый расчёт получает новый ключ.
- **Таймаут по last_modified.** Heartbeat обновляет время каждые 30 секунд. Readiness запускает проверку очереди; отсутствие обновлений 1800 секунд приводит к FAILED. Поток monitor_queue удалён.
- **Shutdown проверяется между GPU-батчами и операциями.** При сигнале остановки под прекращает забирать заявки, а worker сохраняет FAILED с причиной pod_shutdown. Уже начатый вызов GPU или S3 не прерывается мгновенно. При аварийном выключении статус обновит обработчик readiness после таймаута.
- **Публикация идёт по порядку: части результата → _SUCCESS → DONE.** При проблемах сохранения DONE worker повторяет запись статуса, а не весь расчёт. Если проверка таймаута уже выставила FAILED, поздний DONE его не заменяет.
- **Повторы используют те же физические пути.** Airflow очищает только выходной префикс расчёта, сохраняя системный файл очереди. После таймаута остаётся риск поздней записи старого пода в этот путь. Строгая изоляция попыток при текущих ограничениях не гарантируется.
- **История сохраняется.** Readiness удаляет старые завершённые записи только в S3; PG не очищает историю. Ожидающие заявки не теряются из-за возраста.

Диаграммы сохранены как редактируемые исходники Mermaid (.mmd).
