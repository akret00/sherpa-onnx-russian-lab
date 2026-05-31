"""
Модуль разбирает аргументы командной строки и устанавливает значения по умолчанию
"""
import argparse
# import config

def parse_args():
    """Разбирает агрументы командной строки и устанавливает значения по умолчанию, где они есть"""
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

    # ap.set_defaults(**config.config.get_default_args())
    return ap.parse_args()
