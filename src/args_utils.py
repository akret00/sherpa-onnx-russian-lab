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

    ap.add_argument("--provider", default="cpu", help="onnxruntime provider, usually 'cpu'")
    ap.add_argument("--num-threads", type=int, default=1)
    ap.add_argument("--output-dir", help="Путь к папке с с файлами с распознанным текстом")
    ap.add_argument("--no-timestamps", action="store_true",
        help="Запрещает вывод меток времени в распознанный текст")

    # VAD параметры
    ap.add_argument("--vad-threshold", type=float, default=0.4) # 0.4 Было 0.3
    ap.add_argument("--vad-min-silence", type=float, default=0.1) # 0.1 Было 0.25
    ap.add_argument("--vad-min-speech", type=float, default=0.2) # 0.2 Было 0.25
    ap.add_argument("--vad-max-speech", type=float, default=10.0)

    # Embedding параметры для speaker ID
    ap.add_argument("--spk-threshold", type=float, default=0.4,
        help="cosine-sim threshold inside manager.search")
    ap.add_argument("--min-seg-sec", type=float, default=0.5,
        help="Пропускать сегменты короче указанного")

    # Кластеризация
    ap.add_argument("--num-speakers", type=int, default=-1,
        help="Если знаете количество спикеров, задайте его. Иначе оставьте -1" \
            " и установите --cluster-threshold."
    )
    ap.add_argument("--cluster-threshold", type=float, default=0.5,
        help="Используется при --num-speakers=-1. Меньше => больше спикеров," \
            " больше => меньше спикеров."
    )

    # Формат вывода
    ap.add_argument("--pad-sec", type=float, default=0.2,
        help="Защитный интервал для сегментов +/- секунд перед ASR (что бы не обрезать слова).")
    ap.add_argument("--merge-gap", type=float, default=0.25,
        help="Объединять соседние фразы одного спикера, если пауза между ними <= merge-gap.")
    ap.add_argument("--min-turn-sec", type=float, default=0.6,
        help="Пропустить диаризацию фраз, короче, чем --min-turn-sec.")
    ap.add_argument("--show-progress", action="store_true",
        help="Показывать прогресс диаризации")

    args = ap.parse_args()

    # Записываем нужные аргументы в конфигурация пайплайна pl_conf
    if args.provider:
        pl_conf.provider = args.provider
    if args.num_threads:
        pl_conf.num_threads = int(args.num_threads)
    if args.output_dir: # Путь к папке с с файлами с распознанным текстом
        pl_conf.output_dir = args.output_dir
    if args.no_timestamps: # Запрещает вывод меток времени в распознанный текст
        pl_conf.no_timestamps = True
    # VAD параметры
    if args.vad_threshold:
        pl_conf.vad_threshold = float(args.vad_threshold)
    if args.vad_min_silence:
        pl_conf.vad_min_silence = float(args.vad_min_silence)
    if args.vad_min_speech:
        pl_conf.vad_min_speech = float(args.vad_min_speech)
    if args.vad_max_speech:
        pl_conf.vad_max_speech = float(args.vad_max_speech)
    # Embedding параметры для speaker ID
    if args.spk_threshold: # cosine-sim threshold inside manager.search
        pl_conf.spk_threshold = float(args.spk_threshold)
    if args.min_seg_sec: # Пропускать сегменты короче указанного
        pl_conf.min_seg_sec = float(args.min_seg_sec)
    # Кластеризация
    if args.num_speakers: # Если знаете количество спикеров, задайте его. Иначе оставьте -1
        pl_conf.num_speakers = int(args.num_speakers)
    if args.cluster_threshold: # Используется при --num-speakers=-1. Меньше => больше спикеров
        pl_conf.cluster_threshold = float(args.cluster_threshold)
    # Формат вывода
    if args.pad_sec:    # Защитный интервал для сегментов +/- секунд перед ASR
                        # (что бы не обрезать слова)
        pl_conf.pad_sec = float(args.pad_sec)
    if args.merge_gap:  # Объединять соседние фразы одного спикера, если пауза между
                        # ними <= merge-gap
        pl_conf.merge_gap = float(args.merge_gap)
    if args.min_turn_sec: # Пропустить диаризацию фраз, короче, чем --min-turn-sec
        pl_conf.min_turn_sec = float(args.min_turn_sec)
    if args.show_progress: # Показывать прогресс диаризации
        pl_conf.show_progress = True

    return args
