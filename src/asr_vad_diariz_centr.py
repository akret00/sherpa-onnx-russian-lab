#!/usr/bin/env python3
"""
Скрипт распознает речь из аудиофайла, или микрофона в текст с диаризацией через
центроиды эмбеддингов и их обновление
"""
import time
from pathlib import Path
from config import pl_conf, BASE_DIR
import args_utils
import speaker_storage
import common_utils
from pipeline_vad import CentroidDiarizationPipeline

def main():
    """Основная функция"""
    args = args_utils.parse_args()

    # Засекаем время начала инициализации
    start_time = time.perf_counter()

    # Создаем репозитарий для спикеров и загружаем базу спикеров
    db_repo = speaker_storage.VoiceDbRepository()
    speakers = db_repo.load_speakers()

    # Инициализация пайплайна
    pl = CentroidDiarizationPipeline(pl_config = pl_conf, speakers = speakers)

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
            if seg.speaker:
                spk_name = seg.speaker.name
            else:
                spk_name = "Unknown"
            ts_start = common_utils.format_timestamp(seg.start_time)
            ts_end = common_utils.format_timestamp(seg.end_time)
            print(f"[{ts_start}-{ts_end}] {spk_name}: {seg.text}")
            if args.no_timestamps:
                f.write(f"{spk_name}: {seg.text}" + "\n")
            else:
                f.write(f"[{ts_start}-{ts_end}] {spk_name}: {seg.text}" + "\n")

    # Засекаем время окончания распознавания
    end_time = time.perf_counter()
    print(f"Время распознавания: {end_time - start_time:.6f} секунд")

    # Сохраняем обновленную базу спикеров
    db_repo.save_speakers(
        pl.pipeline_result.speakers,
        update_mode = speaker_storage.SpeakerUpdateMode.UPDATE_ALL
    )


if __name__ == "__main__":
    main()
