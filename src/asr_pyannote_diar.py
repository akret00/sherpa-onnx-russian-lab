#!/usr/bin/env python3
"""
Минимальный пайплайн:
  ffmpeg (decode/resample) -> numpy
  -> sherpa-onnx offline diarization (pyannote segmentation + embedding + clustering)
  -> ASR по каждому сегменту -> печать [start-end] SPK_xx: text
"""

import sys
import time
from dataclasses import dataclass
from typing import List

import model_utils
import args_utils
import ffmpeg_utils
from config import pl_conf, SR
import asr_utils


# -----------------------------
# Утилиты времени/сегментов
# -----------------------------

def fmt_ts(seconds: float) -> str:
    """Красивый таймкод  HH:MM:SS.mmm"""
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000.0))
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


@dataclass
class Turn:
    """Содержит данные для фразы одного спикера"""
    start: float
    end: float
    speaker: int


def merge_adjacent_turns(turns: List[Turn], max_gap: float = 0.25) -> List[Turn]:
    """
    Склеиваем соседние сегменты одного и того же speaker_id,
    если между ними небольшой разрыв (часто полезно, чтобы ASR не резал фразы).
    """
    if not turns:
        return []

    turns = sorted(turns, key=lambda t: (t.start, t.end))
    out = [turns[0]]
    for t in turns[1:]:
        last = out[-1]
        if t.speaker == last.speaker and (t.start - last.end) <= max_gap:
            out[-1] = Turn(start=last.start, end=max(last.end, t.end), speaker=last.speaker)
        else:
            out.append(t)
    return out


def clamp(x: int, lo: int, hi: int) -> int:
    """Ограничивает x между lo и hi"""
    return max(lo, min(hi, x))

# -----------------------------
# Sherpa-onnx: diarization + ASR
# -----------------------------
def progress_callback(done: int, total: int) -> int:
    """Коллбэк функция, которая выводит прогресс диаризации"""
    # sherpa-onnx ожидает int return (0 = continue)
    p = 100.0 * done / max(1, total)
    print(f"\rПрогресс диаризации: {p:6.2f}%", end="", file=sys.stderr)
    if done == total:
        print("", file=sys.stderr)
    return 0

def main() -> None:
    """Основная функция"""
    # Засекаем время начала инициализации
    init_start_time = time.perf_counter()

    args = args_utils.parse_args()

    # 1) Create diarizer first: it defines the expected sample rate.
    sd = model_utils.load_pyannote_diarization()

    # 2) Create ASR recognizer once (hot instance).
    recognizer = asr_utils.SherpaASRAdapter(pl_config = pl_conf)

    # Засекаем время окончания инициализации
    init_end_time = time.perf_counter()
    print(f"Время инициализации: {init_end_time - init_start_time:.6f} секунд")

    # 3) Decode input file via ffmpeg directly into float32 mono at sd.sample_rate.
    audio = ffmpeg_utils.read_all_samples(args.input)

    # Засекаем время начала диаризации
    diar_start_time = time.perf_counter()

    # 4) Run diarization.
    print("Начинается диаризация, это может занять много времени...")
    pl_conf.segmentation.show_progress = True
    if pl_conf.segmentation.show_progress:
        diar = sd.process(audio, callback=progress_callback).sort_by_start_time()
    else:
        diar = sd.process(audio).sort_by_start_time()

    # Засекаем время окончания диаризации
    diar_end_time = time.perf_counter()
    print(f"Время диаризации: {diar_end_time - diar_start_time:.6f} секунд")

    turns: List[Turn] = []
    print("Результаты диаризации:")
    for r in diar:
        # dur = r.end - r.start
        # if dur >= pl_conf.segmentation.min_turn_sec:
        turns.append(Turn(start=float(r.start), end=float(r.end), speaker=int(r.speaker)))
        print(f"[{fmt_ts(r.start)} - {fmt_ts(r.end)}] SPK_{r.speaker:02d}")

    # turns = merge_adjacent_turns(turns, max_gap = pl_conf.segmentation.merge_gap)

    # Засекаем время начала распознавания
    asr_start_time = time.perf_counter()

    # 5) ASR per turn and print result with timing + speaker label.
    print("Начинается распознавание речи...")
    pad = pl_conf.segmentation.pad_sec
    audio_len = len(audio)

    for t in turns:
        s = max(0.0, t.start - pad)
        e = min(len(audio) / SR, t.end + pad)

        i0 = clamp(int(s * SR), 0, audio_len)
        i1 = clamp(int(e * SR), 0, audio_len)
        if i1 <= i0:
            continue

        seg_audio = audio[i0:i1]
        text = recognizer.decode_asr(samples_f32 = seg_audio)

        if text:
            print(f"[{fmt_ts(t.start)} - {fmt_ts(t.end)}] SPK_{t.speaker:02d}: {text}")

    # Засекаем время окончания распознавания
    asr_end_time = time.perf_counter()
    print(f"Время распознавания: {asr_end_time - asr_start_time:.6f} секунд")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Важно: даже при Ctrl+C ffmpeg будет корректно прибит через finally в decode_with_ffmpeg...
        print("\nInterrupted.", file=sys.stderr)
        raise
