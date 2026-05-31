"""Управляет подпроцессом ffmpeg"""
import subprocess
import numpy
import config

def make_ffmpeg_proc_for_file(path: str):
    """Создает подпроцесс с ffmpeg"""
    # Decode to 16kHz mono signed 16-bit little-endian PCM to stdout
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", # Умная нормализация громкости
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
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", # Умная нормализация громкости
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
    """Читаем из потока подпроцесса блок данных"""
    window_bytes = window_size * 2  # s16le

    # Пробуем прочитать полный блок данных (0.032 секунды аудио)
    b = read_exactly(proc.stdout, window_bytes)

    # Получаем блок данных в pcm16 формате
    pcm16 = numpy.frombuffer(b, dtype=numpy.int16)
    samples = pcm16.astype(numpy.float32) / 32768.0  # [-1, 1]
    return samples

def read_all_samples(path: str, step_minutes: int = 10):
    """Читаем из потока подпроцесса все данные сразу"""
    # 512 сэмплов при 16кГц — это как раз 0.032 секунды (window_size)
    window_size = 512

    # Считаем размер одного шага увеличения буфера в количестве сэмплов (10 минут)
    # 16000 сэмплов/сек * 60 сек * 10 минут = 9,600,000 сэмплов (~36.6 МБ)
    step_samples = config.SR * 60 * step_minutes

    # Инициализируем массив начальным размером в 1 шаг
    current_capacity = step_samples
    result_array = numpy.zeros(current_capacity, dtype=numpy.float32)
    write_pointer = 0

    proc = make_ffmpeg_proc_for_file(path)

    # Обеспечение закрытия подпроцесса и освобождения ресурсов
    try:
        while True:
            # Вызываем функцию для чтения и конвертации блока
            samples = read_samples(proc, window_size)

            # Если функция вернула пустой массив, значит достигнут конец файла (EOF)
            num_samples = len(samples)
            if num_samples == 0:
                break

            # Если места для нового блока не хватает, увеличиваем массив на 1 шаг
            if write_pointer + num_samples > current_capacity:
                current_capacity += step_samples
                # refcheck=False обязателен, чтобы numpy не ругался на внутренние ссылки
                result_array.resize(current_capacity, refcheck=False)

            # Записываем данные напрямую в выделенную память
            result_array[write_pointer:write_pointer + num_samples] = samples
            write_pointer += num_samples
    finally:
        close_ffmpeg_proc(proc)

    # Обрезаем массив до фактически записанного размера
    result_array.resize(write_pointer, refcheck=False)
    return result_array
