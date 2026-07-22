#!/usr/bin/env python3
"""
Скрипт распознает речь из аудиофайла, или микрофона в текст с диаризацией через
центроиды эмбеддингов и их обновление
"""
import time
import datetime
from pathlib import Path
from config import config, PipelineType, SpeakerRepoType
import args_utils
import common_utils
from pipeline_vad import CentroidDiarizationPipeline

def main() -> None:
    """Основная функция"""
    args = args_utils.parse_args()

    # Засекаем время начала инициализации
    start_time = time.perf_counter()

    # Инициализация пайплайна
    pl_conf = config.get_new_pipeline_config()
    pl_conf.runtime.pipeline_type = PipelineType.CENTRIOD_DIARIZ_PIPELINE
    # pl_conf.diar_vad.speaker_repo_type = SpeakerRepoType.DB_SQLITE # Хранение спикеров в БД
    pl_conf.diar_vad.speaker_repo_type = SpeakerRepoType.IN_MEMORY # Хранение спикеров в памяти
    pl = CentroidDiarizationPipeline(pl_config = pl_conf)

    # Определяем путь к аудио файлу
    if args.mic:
        a_path = "mic"
    else:
        a_path = args.input

    # Определяем директорию для хранения файлов с результатом
    output_dir = common_utils.get_output_dir()
    # Определяем  имя файла с результатом распознавания
    time_str = ""
    if a_path == "mic":
        time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file_path = output_dir / f"{Path(a_path + time_str).name}.txt"
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
            if seg.speaker:
                spk_name = seg.speaker.name
            else:
                spk_name = "Unknown"
            ts_start = common_utils.format_timestamp(seg.start_time)
            ts_end = common_utils.format_timestamp(seg.end_time)
            print(f"[{ts_start}-{ts_end}] {spk_name}: {seg.text}")
            if pl_conf.runtime.no_timestamps:
                f.write(f"{spk_name}: {seg.text}" + "\n")
            else:
                f.write(f"[{ts_start}-{ts_end}] {spk_name}: {seg.text}" + "\n")

    # Засекаем время окончания распознавания
    end_time = time.perf_counter()
    print(f"Время распознавания: {end_time - start_time:.6f} секунд")

if __name__ == "__main__":
    main()
