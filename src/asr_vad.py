#!/usr/bin/env python3
"""
Скрипт распознает речь из аудиофайла, или микрофона в текст без диаризации
"""
import time
from pathlib import Path
from config import pl_conf, BASE_DIR
import args_utils
import common_utils
from pipeline_vad import AsrPipeline

def main():
    """Основная функция"""
    args = args_utils.parse_args()

    # Засекаем время начала инициализации
    start_time = time.perf_counter()

    # Инициализация пайплайна
    pl = AsrPipeline(pl_config = pl_conf)

    # Определяем путь к аудио файлу
    if args.mic:
        a_path = "mic"
    else:
        a_path = args.input

    # Определяем директорию для хранения файлов с результатом
    output_dir = common_utils.get_output_path(args, BASE_DIR)
    # Определяем  имя файла с результатом распознавания
    output_file_path = output_dir / f"{Path(a_path).name}.txt"
    # Создаем директорию, если ее еще нет и открываем файл
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Засекаем время окончания инициализации
    end_time = time.perf_counter()
    print(f"Время инициализации: {end_time - start_time:.6f} секунд")

    # Засекаем время начала распознавания
    start_time = time.perf_counter()

    # Запуск пайплайна обработки аудио
    with open(output_file_path, "w", encoding="utf-8") as f:
        for seg in pl.run_as_stream(a_path):
            ts_start = common_utils.format_timestamp(seg.start_time)
            ts_end = common_utils.format_timestamp(seg.end_time)
            print(f"[{ts_start}-{ts_end}] {seg.text}")
            if pl_conf.no_timestamps:
                f.write(f"{seg.text}" + "\n")
            else:
                f.write(f"[{ts_start}-{ts_end}] {seg.text}" + "\n")

    # Засекаем время окончания распознавания
    end_time = time.perf_counter()
    print(f"Время распознавания: {end_time - start_time:.6f} секунд")


if __name__ == "__main__":
    main()
