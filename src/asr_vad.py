#!/usr/bin/env python3
"""
Скрипт распознает речь из аудиофайла, или микрофона в текст без диаризации
"""
import sys
import time
import numpy as np
import sherpa_onnx
from config import config, SR
import model_utils
import ffmpeg_utils
import asr_utils
import args_utils
import vad_utils

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

    # Создаем подпроцесс с потоком от ffmpeg для микрофона или аудиофайла
    if args.mic:
        proc = ffmpeg_utils.make_ffmpeg_proc_for_pulse_default()
    else:
        proc = ffmpeg_utils.make_ffmpeg_proc_for_file(args.input)
    if proc.stdout is None:
        print("ffmpeg stdout is None", file=sys.stderr)
        ffmpeg_utils.close_ffmpeg_proc(proc)
        sys.exit(1)

    # Засекаем время окончания инициализации
    end_time = time.perf_counter()
    print(f"Время инициализации: {end_time - start_time:.6f} секунд")

    # Обеспечение закрытия подпроцесса и освобождения ресурсов
    try:
        # Засекаем время начала распознавания
        start_time = time.perf_counter()

        # window_size обычно 512, соответствует 0.032 секунды аудио
        window_bytes = window_size * 2  # s16le

        # Оснвной цикл
        while True:
            # Пробуем прочитать полный блок данных (0.032 секунды аудио)
            b = asr_utils.read_exactly(proc.stdout, window_bytes)
            # Если блок пустой или неполный, то игнорируем его и переходим к выталкиванию
            # из VAD тишиной последней незавершенной фразы, если она есть
            if len(b) == 0 or len(b) < window_bytes:
                break

            # Получаем блок данных в pcm16 формате
            pcm16 = np.frombuffer(b, dtype=np.int16)
            samples = pcm16.astype(np.float32) / 32768.0  # [-1, 1]
            # Передает в VAD очередной блок данных. Когда VAD определяет начало фразы, он начинает
            # накапливать фрагменты фразы до тех пор, пока не определит завершение фразы.
            # После этого vad.empty() возвращает False и фразу можно забирать целиком.
            # После извлечения фразы из VAD методом vad.pop(), VAD становится пустым,
            # и vad.empty() == True
            vad.accept_waveform(samples)
            vad_utils.process_vad_segments(vad, recognizer)

        # Проталкиваем в VAD последнюю неоконченную фразу 1 секундой тишины (нулевые данные)
        zeros = np.zeros(window_size, dtype=np.float32)
        for _ in range(int(SR / window_size) + 2):
            vad.accept_waveform(zeros)
            vad_utils.process_vad_segments(vad, recognizer)

        # Засекаем время окончания распознавания
        end_time = time.perf_counter()
        print(f"Время распознавания: {end_time - start_time:.6f} секунд")
    finally:
        ffmpeg_utils.close_ffmpeg_proc(proc)

if __name__ == "__main__":
    main()
