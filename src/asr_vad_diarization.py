#!/usr/bin/env python3
"""
Скрипт распознает речь из аудиофайла, или микрофона в текст с диаризацией через эмбеддинги
"""
import sys
import time
import numpy as np
from config import config, SR
import model_utils
import ffmpeg_utils
import asr_utils
import args_utils
import vad_utils
import diarization_utils
import speaker_storage

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

    # Создаем репозитарий для спикеров
    db_repo = speaker_storage.VoiceDbRepository()

    #Инициализируем распознаватель голоса
    speaker_resolver = diarization_utils.SpeakerResolver(
        num_threads = args.num_threads,
        spk_threshold = args.spk_threshold,
        resolving_mode = diarization_utils.SpeakerResolvingMode.VAD_SPEAKER_MANAGER,
        speakers = db_repo.load_speakers()
    )

    # print(f"Загружено спикеров: {len(speaker_resolver.get_speakers())}")
    # for spk in speaker_resolver.get_speakers():
    #     print(f"ID: {spk.id}  Name: {spk.name}  Total count: {spk.total_count}")

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

        # Оснвной цикл
        while True:
            # Пробуем прочитать полный блок (window_size == 512) данных (0.032 секунды аудио)
            samples = ffmpeg_utils.read_samples(proc, window_size)
            # Если блок пустой или неполный, то игнорируем его и переходим к выталкиванию
            # из VAD тишиной последней незавершенной фразы, если она есть
            if len(samples) == 0 or len(samples) < window_size:
                break

            # Передает в VAD очередной блок данных. Когда VAD определяет начало фразы, он начинает
            # накапливать фрагменты фразы до тех пор, пока не определит завершение фразы.
            # После этого vad.empty() возвращает False и фразу можно забирать целиком.
            # После извлечения фразы из VAD методом vad.pop(), VAD становится пустым,
            # и vad.empty() == True
            vad.accept_waveform(samples)
            for segment, t_start, t_end in vad_utils.get_speec_segments(vad):
                # Распознаем (ASR) полученный из VAD сегмент
                text = asr_utils.decode_asr(recognizer, segment)
                # Распознаем спикера
                speaker_name = speaker_resolver.resolve(segment)

                if text:
                    print(f"[{t_start:10.3f}-{t_end:10.3f}] {speaker_name}: {text}")

        # Проталкиваем в VAD последнюю неоконченную фразу 1 секундой тишины (нулевые данные)
        zeros = np.zeros(window_size, dtype=np.float32)
        for _ in range(int(SR / window_size) + 2):
            vad.accept_waveform(zeros)
            for segment, t_start, t_end in vad_utils.get_speec_segments(vad):
                # Распознаем (ASR) полученный из VAD сегмент
                text = asr_utils.decode_asr(recognizer, segment)
                # Распознаем спикера
                speaker_name = speaker_resolver.resolve(segment)

                if text:
                    print(f"[{t_start:10.3f}-{t_end:10.3f}] {speaker_name}: {text}")

        # Засекаем время окончания распознавания
        end_time = time.perf_counter()
        print(f"Время распознавания: {end_time - start_time:.6f} секунд")
    finally:
        ffmpeg_utils.close_ffmpeg_proc(proc)

    # Сохраняем обновленную базу спикеров
    db_repo.save_speakers(
        speaker_resolver.get_speakers(),
        update_mode = speaker_storage.SpeakerUpdateMode.UPDATE_ALL
    )

if __name__ == "__main__":
    main()
