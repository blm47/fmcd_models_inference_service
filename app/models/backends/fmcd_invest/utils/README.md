# Pipeline INVEST

Перенесён по 27 скриншотам ноутбука от 2026-09-02. Backend принимает Parquet
с исходными признаками; SQL/Hadoop-выгрузка выполняется снаружи сервиса.
Пары `counterparty_id` + `report_dt` должны быть уникальными во всём входном
наборе и не содержать NULL. Backend проверяет это внутри батча; уникальность
между батчами и файлами обеспечивает поставщик данных.

## Размещение артефактов

Положите файлы с исходными именами в следующую структуру. `artifacts_fmcd`
сохранён как вложенный каталог из ноутбука; `artefacts` в корне задан пользователем.

```text
artefacts/fmcd_invest/
  fmcd_reference.xlsx
  mrmr_report_v2.xlsx
  catboost_multilabel_model_v1_optimized.cbm
  calib_model_weights.json
  artifacts_fmcd/
    feature_preprocessor.json
    stacking_training_config.json
    experiments_stacking/
      artifacts/
        fmcd_stacking_best_state.pt
    experiments_single_target/
      F/artifacts/fmcd_nn_artifacts.json
      M/artifacts/fmcd_nn_artifacts.json
      C/artifacts/fmcd_nn_artifacts.json
      D/artifacts/fmcd_nn_artifacts.json
```

Checkpoint должен быть полным `state_dict`, включая `base_models.F/M/C/D.*`.
Отдельные веса базовых сетей не нужны: из четырёх JSON читается
`model_init_params`, затем общий checkpoint загружается со `strict=True`.
Количество и порядок признаков берутся из артефактов, а не из чисел на скриншотах.

## Стадии и результат

1. По `fmcd_reference.xlsx` выбираются типы признаков и колонки zero-fill.
   Заполнение нулём выполняется перед приведением типов, как в Spark-коде.
   В Parquet нужно сохранять исходные типы числовых/строковых признаков.
2. CatBoost получает признаки в `feature_names_`, с токеном `Special: Missings`
   для категориальных пропусков. Выходы 0/1/0/2 соответствуют F/M/C/D.
   Калибровка: `sigmoid(w * log(p / (1 - p)) + w0)`, без clipping вероятностей.
3. Для NN выполняются включённые шаги JSON: заполнение и missing-флаги,
   winsorization, transform, scaling, categorical encoding. Проверяются набор
   колонок, порядок `feature_columns`, числовые типы и отсутствие пропусков.
4. Четыре MLP передают embeddings и предсказания в stacking. После inference
   применяются обратные normalization/transform. Регрессия F ограничивается снизу
   нулём, C — диапазоном 0–62, D — 0–13; M не ограничивается.
5. Калиброванные вероятности умножаются на регрессии. NumPy округляет F/C/D
   до целого, M — до двух знаков. Типы всех скоров — float64.

Выходные 14 колонок в порядке ноутбука:
`counterparty_id`, `report_dt`, `f_prob_score`, `m_prob_score`, `c_prob_score`,
`d_prob_score`, `f_prob_score_calib`, `m_prob_score_calib`, `c_prob_score_calib`,
`d_prob_score_calib`, `f_pred`, `m_pred`, `c_pred`, `d_pred`.
Типы и порядок ключей сохраняются. При согласованной уникальности ключей
построчное объединение веток эквивалентно итоговому inner merge ноутбука.

`device` и размеры батчей задаются в `configs/models.yaml`: для INVEST
по умолчанию CUDA, infer batch 4096, Parquet chunk 5000. CatBoost использует CPU.
Для запуска без GPU укажите `device: "cpu"`. Все сообщения идут через logger сервиса.

## Проверка

`python -m unittest discover -s tests -p test_invest.py` проверяет преобразования,
формулы, загрузку синтетических Excel/JSON/CatBoost/PyTorch артефактов, батчи
и lifecycle. Для полной проверки нужны зависимости из requirements, включая
CatBoost, openpyxl и PyTorch. Без них модельные тесты пропускаются.
Приёмка на настоящих весах требует сравнения промежуточных и итоговых скоров
с эталоном ноутбука; синтетические тесты не подтверждают такую эквивалентность.
