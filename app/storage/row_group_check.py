from typing import Any


def warn_on_oversized_row_groups(dataset, chunk_size: int, logger: Any) -> None:
    """
    Диагностический проход ПЕРЕД чтением через dataset.to_batches().
    Проходит по каждому парт-файлу датасета и проверяет его
    row group'ы через footer-метаданные.
    Если у фрагмента ровно 1 row group и строк в нём больше chunk_size -
    это тревожный признак: to_batches() будет вынужден продекодировать
    этот row group ЦЕЛИКОМ в память перед тем, как отдать из него хоть
    один батч, независимо от значения chunk_size. Явно логируем warn
    с именем файла и точным размером, чтобы при повторном OOM не искать
    причину заново, а сразу понимать, что виноват.
    """
    for fragment in dataset.get_fragments():
        num_row_groups = fragment.num_row_groups
        if num_row_groups == 1:
            row_group_meta = fragment.metadata.row_group(0)
            rg_rows = row_group_meta.num_rows
            rg_bytes = row_group_meta.total_byte_size
            if rg_rows > chunk_size:
                logger.warn(
                    f"Файл {fragment.path}: содержит ВСЕГО 1 row group на {rg_rows} строк "
                    f"(> chunk_size={chunk_size}), объём {rg_bytes / 1e6:.1f} MB. "
                    "dataset.to_batches() будет декодировать этот row group ЦЕЛИКОМ в память "
                    "перед выдачей первого батча из него, независимо от chunk_size. "
                    "Риск OOM пропорционален размеру файла."
                )

        else:
            # Дополнительно проверяем каждый row group на случай, если
            # они у файла неравномерные
            for rg_idx in range(num_row_groups):
                rg_meta = fragment.metadata.row_group(rg_idx)
                if rg_meta.num_rows > chunk_size:
                    logger.warn(
                        f"Файл {fragment.path}: row group {rg_idx}/{num_row_groups} "
                        f"содержит {rg_meta.num_rows} строк (> chunk_size={chunk_size}), "
                        f"объём {rg_meta.total_byte_size / 1e6:.1f} MB — будет "
                        "декодирован целиком за один внутренний проход."
                    )
