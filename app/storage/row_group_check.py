from typing import Any


def warn_on_oversized_row_groups(dataset, chunk_size: int, logger: Any) -> None:
    """
    Проверяет размеры row groups по footer-метаданным до чтения.
    Большие row groups могут увеличивать память на декодирование и prefetch.
    chunk_size ограничивает размер выдаваемого батча, но не общий пик RAM;
    фактический расход зависит от схемы, кодирования и настроек PyArrow.
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
                    "Размер батча не ограничивает общий пик памяти reader; "
                    "проверьте расход RAM на декодирование и prefetch."
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
                        f"объём {rg_meta.total_byte_size / 1e6:.1f} MB — "
                        "возможен повышенный расход памяти reader."
                    )
