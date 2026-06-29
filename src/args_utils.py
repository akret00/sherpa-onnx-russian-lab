"""
Модуль разбирает аргументы командной строки и устанавливает значения по умолчанию
"""
import argparse
from config import pl_conf

def parse_args():
    """
    Разбирает агрументы командной строки и устанавливает значения по умолчанию, где они есть
    Так же, модифицирует значения параметров в pl_conf
    """
    ap = argparse.ArgumentParser()

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="Путь к аудио файлу (mp3/wav/...)")
    src.add_argument("--mic", action="store_true",
        help="Чтение с микрофона через PulseAudio (default source)")

    ap.add_argument("--provider", help="onnxruntime provider, usually 'cpu'")
    ap.add_argument("--num-threads", type=int)
    ap.add_argument("--output-dir", help="Путь к папке с с файлами с распознанным текстом")
    ap.add_argument("--no-timestamps", action="store_true",
        help="Запрещает вывод меток времени в распознанный текст")

    # VAD параметры
    ap.add_argument("--vad-threshold", type=float)
    ap.add_argument("--vad-min-silence", type=float)
    ap.add_argument("--vad-min-speech", type=float)
    ap.add_argument("--vad-max-speech", type=float)

    # Embedding параметры для speaker ID
    ap.add_argument("--spk-threshold", type=float,
        help="cosine-sim threshold inside manager.search")
    ap.add_argument("--min-seg-sec", type=float,
        help="Пропускать сегменты короче указанного")

    # Кластеризация
    ap.add_argument("--num-speakers", type=int,
        help="Если знаете количество спикеров, задайте его. Иначе оставьте -1" \
            " и установите --cluster-threshold."
    )
    ap.add_argument("--cluster-threshold", type=float,
        help="Используется при --num-speakers=-1. Меньше => больше спикеров," \
            " больше => меньше спикеров."
    )

    # Формат вывода
    ap.add_argument("--pad-sec", type=float,
        help="Защитный интервал для сегментов +/- секунд перед ASR (что бы не обрезать слова).")
    ap.add_argument("--merge-gap", type=float,
        help="Объединять соседние фразы одного спикера, если пауза между ними <= merge-gap.")
    ap.add_argument("--min-turn-sec", type=float,
        help="Пропустить диаризацию фраз, короче, чем --min-turn-sec.")
    ap.add_argument("--show-progress", action="store_true",
        help="Показывать прогресс диаризации")

    args = ap.parse_args()

    # Записываем нужные аргументы в конфигурацию пайплайна pl_conf
    if args.provider:
        pl_conf.runtime.provider = args.provider
    if args.num_threads:
        pl_conf.runtime.num_threads = int(args.num_threads)
    if args.output_dir: # Путь к папке с с файлами с распознанным текстом
        pl_conf.runtime.output_dir = args.output_dir
    if args.no_timestamps: # Запрещает вывод меток времени в распознанный текст
        pl_conf.runtime.no_timestamps = True
    # VAD параметры
    if args.vad_threshold:
        pl_conf.vad.threshold = float(args.vad_threshold)
    if args.vad_min_silence:
        pl_conf.vad.min_silence = float(args.vad_min_silence)
    if args.vad_min_speech:
        pl_conf.vad.min_speech = float(args.vad_min_speech)
    if args.vad_max_speech:
        pl_conf.vad.max_speech = float(args.vad_max_speech)
    # Embedding параметры для speaker ID
    if args.spk_threshold: # cosine-sim threshold inside manager.search
        pl_conf.segmentation.spk_threshold = float(args.spk_threshold)
    if args.min_seg_sec: # Пропускать сегменты короче указанного
        pl_conf.segmentation.min_seg_sec = float(args.min_seg_sec)
    # Кластеризация
    if args.num_speakers: # Если знаете количество спикеров, задайте его. Иначе оставьте -1
        pl_conf.segmentation.num_speakers = int(args.num_speakers)
    if args.cluster_threshold: # Используется при --num-speakers=-1. Меньше => больше спикеров
        pl_conf.segmentation.cluster_threshold = float(args.cluster_threshold)
    # Формат вывода
    if args.pad_sec:    # Защитный интервал для сегментов +/- секунд перед ASR
                        # (что бы не обрезать слова)
        pl_conf.segmentation.pad_sec = float(args.pad_sec)
    if args.merge_gap:  # Объединять соседние фразы одного спикера, если пауза между
                        # ними <= merge-gap
        pl_conf.segmentation.merge_gap = float(args.merge_gap)
    if args.min_turn_sec: # Пропустить диаризацию фраз, короче, чем --min-turn-sec
        pl_conf.segmentation.min_turn_sec = float(args.min_turn_sec)
    if args.show_progress: # Показывать прогресс диаризации
        pl_conf.segmentation.show_progress = True

    return args
