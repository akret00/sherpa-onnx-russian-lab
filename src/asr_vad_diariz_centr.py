#!/usr/bin/env python3
"""
Скрипт распознает речь из аудиофайла, или микрофона в текст с диаризацией через
центроиды эмбеддингов и их обновление
"""
import time
from config import pl_conf
import args_utils
import speaker_storage
from pipeline_centroid import CentroidDiarizationPipeline

def main():
    """Основная функция"""
    args = args_utils.parse_args()

    # Засекаем время начала инициализации
    start_time = time.perf_counter()

    # Создаем репозитарий для спикеров и загружаем базу спикеров
    db_repo = speaker_storage.VoiceDbRepository()
    speakers = db_repo.load_speakers()

    # Инициализация пайплайна
    if args.mic:
        a_path = "mic"
    else:
        a_path = args.input
    pl = CentroidDiarizationPipeline(pl_config = pl_conf, speakers = speakers)

    # Засекаем время окончания инициализации
    end_time = time.perf_counter()
    print(f"Время инициализации: {end_time - start_time:.6f} секунд")

    # Засекаем время начала распознавания
    start_time = time.perf_counter()

    # Запуск пайплайна обработки аудио
    pl_result = pl.run(a_path)

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
