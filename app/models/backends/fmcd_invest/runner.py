"""
Точка подключения INVEST до получения полного комплекта артефактов и эталона.
"""

from app.models.contracts import ModelBundle


class FMCDInvestRunner(ModelBundle):
    def load(self) -> None:
        raise NotImplementedError(
            "fmcd_invest: pipeline ещё не реализован; требуются артефакты, "
            "SQL/ноутбук и эталон для проверки результатов"
        )

    def predict_batch(self, frame, check_shutdown):
        raise RuntimeError("fmcd_invest: pipeline не загружен")

    def close(self) -> None:
        """
        Заглушка не создаёт ресурсов; повторное закрытие безопасно.
        """
