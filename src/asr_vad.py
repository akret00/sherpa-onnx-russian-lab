#!/usr/bin/env python3
"""
Скрипт распознает речь из аудиофайла, или микрофона в текст без диаризации
"""
import sys
import time
from pathlib import Path
import numpy as np
from config import config, SR, BASE_DIR, DEFAULT_OUTPUT_DIR
import model_utils
import ffmpeg_utils
import asr_utils
import args_utils
import vad_utils

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

def main():
    """Основная функция"""
    args = args_utils.parse_args()

    # Засекаем время начала инициализации
    start_time = time.perf_counter()

    # Инициализируем VAD
    vad, window_size = model_utils.load_vad(
        vad_model = config.get_vad_model()['model'],
        threshold=args.vad_threshold,
        min_silence=args.vad_min_silence,
        min_speech=args.vad_min_speech,
        max_speech=args.vad_max_speech,
    )

    # Инициализируем ASR распознаватель
    recognizer = model_utils.load_asr(num_threads = args.num_threads, provider = args.provider)

    # Засекаем время окончания инициализации
    end_time = time.perf_counter()
    print(f"Время инициализации: {end_time - start_time:.6f} секунд")

    # Определяем директорию для хранения файлов с результатом
    output_dir = get_output_path(args, BASE_DIR)
    # Определяем  имя файла с результатом распознавания
    output_file_path = output_dir / f"{Path(args.input).name}.txt"

    # Создаем директорию, если ее еще нет и открываем файл
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        # Создаем подпроцесс с потоком от ffmpeg для микрофона или аудиофайла
        if args.mic:
            proc = ffmpeg_utils.make_ffmpeg_proc_for_pulse_default()
        else:
            proc = ffmpeg_utils.make_ffmpeg_proc_for_file(args.input)
        if proc.stdout is None:
            print("ffmpeg stdout is None", file=sys.stderr)
            ffmpeg_utils.close_ffmpeg_proc(proc)
            sys.exit(1)

        # Обеспечение закрытия подпроцесса и освобождения ресурсов
        try:
            # Засекаем время начала распознавания
            start_time = time.perf_counter()

            # Оснвной цикл
            while True:
                # Пробуем прочитать полный блок (window_size == 512) данных (0.032 секунды аудио)
                samples = ffmpeg_utils.read_samples(proc, window_size)
                # Если блок пустой или неполный, то игнорируем его и переходим к выталкиванию
                # из VAD тишиной последней незавершенной фразы, если она есть
                if len(samples) == 0 or len(samples) < window_size:
                    break

                # Передает в VAD очередной блок данных. Когда VAD определяет начало фразы,
                # он начинает накапливать фрагменты фразы до тех пор, пока не определит
                # завершение фразы.
                # После этого vad.empty() возвращает False и фразу можно забирать целиком.
                # После извлечения фразы из VAD методом vad.pop(), VAD становится пустым,
                # и vad.empty() == True
                vad.accept_waveform(samples)
                for segment, t_start, t_end in vad_utils.get_speec_segments(vad):
                    # Распознаем (ASR) полученный из VAD сегмент
                    text = asr_utils.decode_asr(recognizer, segment)

                    if text:
                        print(f"[{t_start:10.3f}-{t_end:10.3f}]: {text}")
                        if args.no_timestamps:
                            f.write(f"{text}" + "\n")
                        else:
                            f.write(f"[{t_start:10.3f}-{t_end:10.3f}]: {text}" + "\n")

            # Проталкиваем в VAD последнюю неоконченную фразу 1 секундой тишины (нулевые данные)
            zeros = np.zeros(window_size, dtype=np.float32)
            for _ in range(int(SR / window_size) + 2):
                vad.accept_waveform(zeros)
                for segment, t_start, t_end in vad_utils.get_speec_segments(vad):
                    # Распознаем (ASR) полученный из VAD сегмент
                    text = asr_utils.decode_asr(recognizer, segment)

                    if text:
                        print(f"[{t_start:10.3f}-{t_end:10.3f}]: {text}")
                        if args.no_timestamps:
                            f.write(f"{text}" + "\n")
                        else:
                            f.write(f"[{t_start:10.3f}-{t_end:10.3f}]: {text}" + "\n")

            # Засекаем время окончания распознавания
            end_time = time.perf_counter()
            print(f"Время распознавания: {end_time - start_time:.6f} секунд")
        finally:
            ffmpeg_utils.close_ffmpeg_proc(proc)

if __name__ == "__main__":
    main()
