"""Управляет подпроцессом ffmpeg"""
import subprocess
from pathlib import Path
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

def read_samples(proc, window_size) -> numpy.ndarray[numpy.float32]:
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


def convert_wav_to_opus(wav_path: Path, opus_path: Path | None = None) -> Path:
    """Конвертирует WAV-файл в формат OPUS с автоматическим управлением ресурсами.

    Гарантированно закрывает подпроцесс FFmpeg и освобождает ресурсы ОС
    даже в случае системных сбоев или некорректных аудиоданных.

    Args:
        wav_path: Путь к исходному WAV-файлу.
        opus_path: Необязательный путь к итоговому .opus файлу.
            Если не указан, создается в той же папке с заменой расширения.

    Returns:
        Path: Путь к созданному файлу .opus.
    """
    if not wav_path.exists():
        raise FileNotFoundError(f"Исходный WAV-файл не найден: {wav_path}")

    # 1. Если выходной путь не передан, заменяем расширение исходного файла на .opus
    if opus_path is None:
        opus_path = wav_path.with_suffix(".opus")

    # Гарантируем, что целевая папка для OPUS существует
    opus_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Формируем аргументы для запуска FFmpeg
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # Автоматически перезаписывать выходной файл, если он есть
        "-i",
        str(wav_path),  # Входной файл
        "-c:a",
        "libopus",  # Официальный кодек Opus
        "-ar",
        "16000",  # Целевая частота дискретизации для бенчмарка
        str(opus_path),  # Выходной файл
    ]

    print(f"Запуск FFmpeg: {wav_path.name} -> {opus_path.name}")

    try:
        # 3. Менеджер контекста 'with' гарантирует вызов методов очистки процесса.
        # Перенаправляем потоки вывода в DEVNULL, чтобы избежать переполнения
        # системных буферов (из-за чего процесс может намертво зависнуть).
        with subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) as process:
            try:
                # Ожидаем завершения кодирования. Метод communicate() гарантирует
                # закрытие стандартных потоков ввода-вывода подпроцесса.
                # timeout=60 защищает от бесконечного зависания, если FFmpeg застрял.
                process.communicate(timeout=60)

            except subprocess.TimeoutExpired as exc:
                # Если FFmpeg завис и превысил таймаут — принудительно уничтожаем его
                process.kill()
                # Даем операционной системе очистить дескрипторы после уничтожения
                process.communicate()
                raise TimeoutError(
                    f"FFmpeg превысил лимит времени ожидания при обработке {wav_path}"
                ) from exc

            except Exception:
                # При любом другом внутреннем исключении (например, Ctrl+C от пользователя)
                process.kill()
                process.communicate()
                raise

            # Проверяем код возврата утилиты после штатного выхода из communicate()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, ffmpeg_cmd
                )

    except FileNotFoundError as exc:
        raise RuntimeError(
            "Утилита 'ffmpeg' не найдена в системе. "
            "Убедитесь, что FFmpeg установлен и добавлен в переменную окружения PATH."
        ) from exc

    return opus_path
