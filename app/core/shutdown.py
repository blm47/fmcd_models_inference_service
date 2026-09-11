"""Передача сигнала остановки worker без подмены штатного shutdown Uvicorn."""

import signal
import threading


def install_shutdown_handlers(stop, logger):
    """Оборачивает handlers Uvicorn, чтобы worker увидел SIGTERM до ожидания HTTP."""
    if threading.current_thread() is not threading.main_thread():
        return {}
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}

    def handle(signum, frame):
        stop.set()
        logger.info(f"Получен сигнал {signum}: новые расчёты не запускаются")
        handler = previous[signum]
        if callable(handler):
            handler(signum, frame)

    for signum in previous:
        signal.signal(signum, handle)
    return previous


def restore_shutdown_handlers(previous):
    for signum, handler in previous.items():
        signal.signal(signum, handler)
