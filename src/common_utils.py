"""Модуль для общих утилит"""
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from config import DEFAULT_OUTPUT_DIR

def get_output_path(args, base_dir: Path) -> Path:
    """Определяет путь к папке для хранения файлов с распознанным текстом"""
    # Если аргумент не передан (None), берем папку по умолчанию
    raw_val = args.output_dir if args.output_dir else DEFAULT_OUTPUT_DIR

    raw_path = Path(raw_val)

    if raw_path.is_absolute():
        # Если это DEFAULT_OUTPUT_DIR, он уже абсолютный (от BASE_DIR)
        return raw_path
    # Если это относительный путь от пользователя — приклеиваем к BASE_DIR
    return base_dir / raw_path

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
