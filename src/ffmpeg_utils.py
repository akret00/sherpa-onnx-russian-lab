"""Управляет подпроцессом ffmpeg"""
import subprocess
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
