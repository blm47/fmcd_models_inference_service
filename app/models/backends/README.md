# Движки применения моделей

В каждом backend — один файл с классом-наследником ModelBundle и каталог utils.
Название ModelRunner обозначает исполнителя модели: загрузка, применение и закрытие.
Другие варианты для обсуждения: ModelEngine (движок), InferenceModel (модель инференса).

```text
backends/
  fmcd_cc_dc/
    runner.py                 # FMCDCardsRunner(ModelBundle)
    utils/
      inference.py            # Вычисления батча
      pandas_to_fmcd.py       # Preprocessing
      fmcd/                  # Полный код моделистов: data/, model/ и т.д.
      requirements.txt
      requirements-container.txt
  fmcd_invest/
    runner.py                 # FMCDInvestRunner(ModelBundle), пока заглушка
    utils/                    # Все дополнительные классы и функции этого движка
```

`__init__.py` только обозначают Python-пакеты. Все дополнительные классы,
функции загрузки, валидации и preprocessing конкретного движка размещаются
в utils или его подпакетах. Общая инфраструктура остаётся вне backend.
Код моделистов использует прямые импорты utils.fmcd, без aliases старого пакета.
Веса и обученные статистики размещаются отдельно в artifacts_dir.

requirements.txt содержит зависимости сверх общей инфраструктуры, а
requirements-container.txt — сохранённые версии корпоративного образа.
Зависимости ещё не предоставленного кода моделистов нужно сверить при его переносе.
