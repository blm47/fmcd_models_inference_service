"""
Общее разбиение чанка на батчи и проверка контракта результата pipeline.
"""

import pandas as pd


def run_inference_on_chunk(df_chunk, bundle, infer_batch_size, check_shutdown):
    """
    Накапливает только результаты текущего чанка; бизнес-ключи не зависят от index.
    """
    results = []
    expected_columns = list(bundle.spec.id_cols) + list(bundle.output_columns)
    expected_dtypes = None
    for start in range(0, len(df_chunk), infer_batch_size):
        check_shutdown()
        original = df_chunk.iloc[start : start + infer_batch_size]
        frame = original.copy(deep=True)
        result = bundle.predict_batch(frame, check_shutdown)
        check_shutdown()
        if not frame.equals(original):
            raise ValueError("Pipeline изменил входной батч")
        if not isinstance(result, pd.DataFrame) or len(result) != len(original):
            raise ValueError("Pipeline должен вернуть DataFrame с одной строкой на входную")
        if list(result.columns) != expected_columns:
            raise ValueError("Колонки результата не соответствуют контракту pipeline")
        keys = list(bundle.spec.id_cols)
        if not result[keys].reset_index(drop=True).equals(original[keys].reset_index(drop=True)):
            raise ValueError("Pipeline изменил значения, типы или порядок ключей")
        dtypes = tuple(result.dtypes)
        if expected_dtypes is not None and dtypes != expected_dtypes:
            raise ValueError("Типы результата изменились между infer-батчами")
        expected_dtypes = dtypes
        results.append(result)
    return pd.concat(results, ignore_index=True)
