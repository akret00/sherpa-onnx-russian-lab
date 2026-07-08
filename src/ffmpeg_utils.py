"""Управляет подпроцессом ffmpeg"""
import subprocess
from pathlib import Path
from typing import IO
from types import TracebackType
from collections.abc import Generator
import numpy
from numpy.typing import NDArray
import config

class AudioStreamReader:
    """Контекстный менеджер для безопасного управления подпроцессом FFmpeg с поддержкой mypy."""
    def __init__(self, path: str, chunk_size: int = 512, duration_sec: float = 0.0) -> None:
        """
        path - путь к аудиофайлу
        chunk_size - размер чанка в сэмплах
        """
        self.path = path
        self.chunk_size: int = chunk_size
        self.process: subprocess.Popen[bytes] | None = None
        self.duration_sec: float = duration_sec
        self.test_source: SilenceAudioSource | None = None

    def __enter__(self) -> "AudioStreamReader":
        """Открываем поток при входе в with, а не при создании объекта"""
        if self.path == config.AUDIO_PATH_ORACLE_EMPTY:
            # Создаем генератор пустых сэмплов
            self.test_source = SilenceAudioSource(duration_sec = self.duration_sec)
        elif self.path == "mic":
            # Создаем подароцесс для чтения с микрофона
            self.process = make_ffmpeg_proc_for_pulse_default()
        else:
            # Создаем подпроцесс для чтения из файла
            self.process = make_ffmpeg_proc_for_file(self.path)
        return self

    def iter_chunks(self) -> Generator[NDArray[numpy.float32], None, None]:
        """Генератор, лениво читающий чанки аудиоданных."""
        if (not self.process or not self.process.stdout) and (not self.test_source):
            raise RuntimeError("Поток не запущен. Используйте менеджер контекста 'with'.")

        while True:
            if self.process: # Если режим чтения аудиоданных
                chunk = read_samples(self.process, self.chunk_size)
            elif self.test_source: # Если режим генерации тестовых данных
                chunk = self.test_source.read_chunk(self.chunk_size)
            else:
                raise ValueError("Оба источника аудиоданных имеют значение None")

            yield chunk

            if len(chunk) < self.chunk_size:
                break

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None
    ) -> bool | None:
        # Если процесс открыт, закрываем его
        if self.process:
            close_ffmpeg_proc(self.process)
        return None


class SilenceAudioSource:
    """Генератор пустых сэмплов (тишины) заданной длительности для тестирования."""
    def __init__(self, duration_sec: float, sample_rate: int = 16000) -> None:
        self.sample_rate: int = sample_rate
        # Вычисляем общее количество сэмплов, которое нужно сгенерировать
        self.total_samples_limit: int = int(duration_sec * sample_rate)
        self.generated_samples_count: int = 0

        # Заранее создаем один пустой массив нужного типа для mypy
        self._empty_result: NDArray[numpy.float32] = numpy.empty(0, dtype=numpy.float32)

    def read_chunk(self, count: int) -> NDArray[numpy.float32]:
        """
        Возвращает массив нулевых сэмплов float32 длиной count.
        Если аудиопоток завершился, возвращает пустой массив.
        """
        # Проверяем, не исчерпан ли лимит сэмплов
        if self.generated_samples_count >= self.total_samples_limit:
            return self._empty_result

        # Если до конца потока осталось меньше сэмплов, чем запрошено (count)
        samples_left: int = self.total_samples_limit - self.generated_samples_count
        current_chunk_size: int = min(count, samples_left)

        # Обновляем счетчик сгенерированных сэмплов
        self.generated_samples_count += current_chunk_size

        # Генерируем массив нулей нужного размера
        return numpy.zeros(current_chunk_size, dtype=numpy.float32)


# ToDo: решить, нужно ли буферизованное чтение из ffmpeg большими блоками?
# Измерить разницу с вариантом без буфера.
class AudioPipeBuffer:
    """Буферизированный загрузчик аудио из потока ffmpeg"""
    def __init__(
        self, ffmpeg_proc: subprocess.Popen[bytes], internal_buff_sec: float = 10.0
    ) -> None:
        """Инициализация объекта класса"""
        self.ffmpeg_proc: subprocess.Popen[bytes] = ffmpeg_proc
        if ffmpeg_proc.stdout is None:
            raise ValueError("ffmpeg_proc.stdout не может быть None")
        self.stdout: IO[bytes] = ffmpeg_proc.stdout

        # 16000 Гц * 2 байта (int16) * internal_chunk_sec
        self.read_size_bytes: int = int(16000 * 2 * internal_buff_sec)

        # Явно объявляем пустой массив нужного типа для mypy
        self.buffer: NDArray[numpy.int16] = numpy.array([], dtype=numpy.int16)

    def get_samples_f32(self, count: int) -> NDArray[numpy.float32]:
        """
        Возвращает ровно count сэмплов в формате float32 [-1.0, 1.0].
        Возвращает None, если поток аудио полностью завершен.
        """
        # Дочитываем данные большими порциями, пока в буфере не наберется нужный count
        while len(self.buffer) < count:
            raw_data: bytes = self.stdout.read(self.read_size_bytes)

            if not raw_data:
                if len(self.buffer) > 0:
                    break
                return numpy.empty(0, dtype=numpy.float32)  # Аудиопоток закончился

            # Быстрое создание массива из байт
            new_samples: NDArray[numpy.int16] = numpy.frombuffer(raw_data, dtype=numpy.int16)
            self.buffer = numpy.append(self.buffer, new_samples)

        # Вырезаем нужный кусок
        output_int16: NDArray[numpy.int16] = self.buffer[:count]
        self.buffer = self.buffer[count:]

        # Дополняем нулями последний чанк, если он меньше запрошенного count
        if len(output_int16) < count:
            output_int16 = numpy.pad(output_int16, (0, count - len(output_int16)))

        # Быстрая и безопасная конвертация int16 [-32768, 32767] в float32 [-1.0, 1.0]
        output_f32: NDArray[numpy.float32] = output_int16.astype(numpy.float32) / 32768.0

        return output_f32

# ToDo: Возможно, стоит убрать методы внутрь соответствующих классов
def make_ffmpeg_proc_for_file(path: str) -> subprocess.Popen[bytes]:
    """Создает подпроцесс с ffmpeg"""
    # Decode to 16kHz mono signed 16-bit little-endian PCM to stdout
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-i", path,
        # "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", # Умная нормализация громкости
        "-ac", "1",
        "-ar", str(config.SR),
        "-f", "s16le",
        "pipe:1",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def make_ffmpeg_proc_for_pulse_default() -> subprocess.Popen[bytes]:
    """Создает подпроцесс с ffmpeg с микрофона по умолчанию"""
    # Microphone via PulseAudio/PipeWire-Pulse "default" source
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-f", "pulse",
        "-i", "default",
        # "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", # Умная нормализация громкости
        "-ac", "1",
        "-ar", str(config.SR),
        "-f", "s16le",
        "pipe:1",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def close_ffmpeg_proc(proc: subprocess.Popen[bytes]) -> None:
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

def read_exactly(stream: IO[bytes], n: int) -> bytes:
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

def read_samples(proc: subprocess.Popen[bytes], window_size: int) -> numpy.ndarray:
    """Читаем из потока подпроцесса блок данных в формате numpy.ndarray[numpy.float32]"""
    window_bytes = window_size * 2  # s16le

    stdout = proc.stdout
    if stdout is None:
        raise ValueError("proc.stdout не может быть None")
    # Пробуем прочитать полный блок данных (0.032 секунды аудио)
    b = read_exactly(stdout, window_bytes)

    # Получаем блок данных в pcm16 формате
    pcm16 = numpy.frombuffer(b, dtype=numpy.int16)
    samples = pcm16.astype(numpy.float32) / 32768.0  # [-1, 1]
    return samples

# ToDo: переместить метод в соответсвующий класс
def read_all_samples(path: str, step_minutes: int = 10) -> numpy.ndarray:
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
