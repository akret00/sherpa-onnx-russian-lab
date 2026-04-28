"""Управляет подпроцессом ffmpeg"""
import subprocess
import numpy as np
import config

def make_ffmpeg_proc_for_file(path: str):
    """Создает подпроцесс с ffmpeg"""
    # Decode to 16kHz mono signed 16-bit little-endian PCM to stdout
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-ac", "1",
        "-ar", str(config.SR),
        "-f", "s16le",
        "pipe:1",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def make_ffmpeg_proc_for_pulse_default():
    """Создает подпроцесс с ffmpeg"""
    # Microphone via PulseAudio/PipeWire-Pulse "default" source
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-f", "pulse",
        "-i", "default",
        "-ac", "1",
        "-ar", str(config.SR),
        "-f", "s16le",
        "pipe:1",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def close_ffmpeg_proc(proc: subprocess.Popen):
    """Закрывает процесс ffmpeg"""
    # Закрываем наши концы pipe, чтобы избежать deadlock
    if proc.stdout:
        proc.stdout.close()
    if proc.stderr:
        proc.stderr.close()
    # Принудительно завершаем процесс
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

def read_exactly(stream, n: int) -> bytes:
    """Read exactly n bytes unless EOF."""
    chunks = []
    got = 0
    while got < n:
        b = stream.read(n - got)
        if not b:
            break
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)

def read_samples(proc, window_size):
    """Читаем из потока подпроцесса данные"""
    window_bytes = window_size * 2  # s16le

    # Пробуем прочитать полный блок данных (0.032 секунды аудио)
    b = read_exactly(proc.stdout, window_bytes)

    # Получаем блок данных в pcm16 формате
    pcm16 = np.frombuffer(b, dtype=np.int16)
    samples = pcm16.astype(np.float32) / 32768.0  # [-1, 1]
    return samples
