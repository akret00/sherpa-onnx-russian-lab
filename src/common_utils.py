"""Модуль для общих утилит"""
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from config import DEFAULT_OUTPUT_DIR, BASE_DIR, pl_conf

def get_output_dir() -> Path:
    """Определяет путь к папке для хранения файлов с распознанным текстом"""
    raw_val = pl_conf.runtime.output_dir or DEFAULT_OUTPUT_DIR
    return (BASE_DIR / raw_val).resolve()

def format_timestamp(seconds: float) -> str:
    """Красивый таймкод  HH:MM:SS.mmm"""
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000.0))
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def get_package_version(package_name: str) -> str | None:
    """
    Функция возвращает версию пакета по его имени, без его импорта.
    Если пакет не найден, возвращается None
    """
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None
