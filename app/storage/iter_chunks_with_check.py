def iter_chunks(self, s3_prefix: str, chunk_size: int):
    """
    Генератор df_chunk по chunk_size строк через dataset.to_batches()
    (оставлено как основной способ чтения).

    Перед стартом чтения выполняется дешёвая диагностика по footer-
    метаданным всех part-файлов: если у какого-то файла row group всего
    один и в нём строк больше chunk_size, логируется WARNING - это
    заранее предупреждает о риске OOM на этом файле, т.к. to_batches()
    будет вынужден продекодировать такой row group целиком в память,
    прежде чем отдать из него хоть один батч.
    """
    dataset = self.open_dataset(s3_prefix)

    warn_on_oversized_row_groups(dataset, chunk_size)

    for batch in dataset.to_batches(batch_size=chunk_size):
        yield batch.to_pandas()
