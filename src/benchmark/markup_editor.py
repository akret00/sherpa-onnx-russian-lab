"""Модуль редактирует разметку аудиофайлов на сегменты и позволяет прослушивать сегменты"""
# Запуск: PYTHONPATH=src python src/benchmark/markup_editor.py --input yaml_file_path
import subprocess
import os
import platform
import numpy
import sounddevice
from benchmark.dataset_entities import AudioSegmentMarkup
import args_utils
from benchmark.dataset_storage import load_markup_from_yaml, export_markup_to_yaml
import ffmpeg_utils
from config import SR

class AudioSegmentEditor:
    """Обеспечивает консольный редактор аудиосегментов"""
    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        # Загружаем спикеров и аудиофайл из YAML
        self.load_yaml()
        self.index = 0
        self.step = 0.05  # Шаг сдвига границ по умолчанию (50 мс)
        self.audio_data: numpy.ndarray | None = None

    @property
    def current(self) -> AudioSegmentMarkup:
        """Возвращает текущий сегмент аудио"""
        return self.segments[self.index]

    def load_yaml(self):
        """Загружает данные из yaml файла с разметкой"""
        self.speakers, self.audio_file = load_markup_from_yaml(self.yaml_path)
        self.segments = sorted(self.audio_file.segments, key=lambda x: x.start_time)
        self.audio_file.segments = self.segments
        print("Данные загружены из файла: {self.yaml_path}")

    def load_audio_data(self, file_path: str):
        """Загружает аудиофайл в формате PCM, 16 кГц, моно"""
        self.audio_data = ffmpeg_utils.read_all_samples(file_path)

    def play_segment_sounddevice(self):
        """Воспроизводит аудиофрагмент с помощью sounddevice."""
        seg = self.current

        # Считаем, что все сегменты с разметкой относятся только к одному файлу
        # Поэтому берем путь к аудиофайлу из первого сегмента и загружаем аудиоданные
        # Позже это стоит отрефакторить, начиная с подхода к процессу разметки
        if self.audio_data is None:
            if not seg.audio_file or not seg.audio_file.file_path:
                print("Ошибка: Нет пути к аудиофайлу.")
                return

            # Проверка наличия аудиофайла
            if not os.path.exists(seg.audio_file.file_path):
                print(f"Ошибка: Файл '{seg.audio_file.file_path}' не существует!")

            print("Начинаем загрузку аудиофайла. Загрузка будет производиться только один раз.")
            self.load_audio_data(seg.audio_file.file_path)

        duration = seg.end_time - seg.start_time
        if duration <= 0:
            print("Ошибка: Невалидная длительность сегмента.")
            return

        # Переводим секунды в индексы массива (сэмплы)
        start_sample = int(seg.start_time * SR)
        end_sample = int(seg.end_time * SR)

        # Защита от выхода за границы массива
        start_sample = max(0, min(start_sample, len(self.audio_data)))
        end_sample = max(0, min(end_sample, len(self.audio_data)))

        if start_sample >= end_sample:
            print("Ошибка: Нулевая или отрицательная длительность среза.")
            return

        # Вырезаем точный кусок без копирования памяти (обычный slice)
        segment_slice = self.audio_data[start_sample:end_sample]

        print(
            f"Воспроизведение: {seg.start_time:.2f}s -> {seg.end_time:.2f}s "
            f"({end_sample - start_sample} сэмплов)"
        )

        # Воспроизводим массив напрямую в звуковую карту
        sounddevice.play(segment_slice, samplerate=SR)
        # Ждем окончания проигрывания текущего фрагмента
        sounddevice.wait()

    def play_segment_ffplay(self):
        """Воспроизводит аудиофрагмент с помощью ffplay."""
        seg = self.current
        if not seg.audio_file or not seg.audio_file.file_path:
            print("Ошибка: Нет пути к аудиофайлу.")
            return

        duration = seg.end_time - seg.start_time
        if duration <= 0:
            print("Ошибка: Невалидная длительность сегмента.")
            return

        # Проверка наличия аудиофайла
        if not os.path.exists(seg.audio_file.file_path):
            print(f"Ошибка: Файл '{seg.audio_file.file_path}' не существует!")

        # -ss: старт, -t: длительность, -nodisp: без окна визуализации,
        # -autoexit: закрыть по окончании
        cmd = [
            "ffplay", "-ss", str(seg.start_time), 
            "-t", str(duration), "-nodisp", "-autoexit", 
            seg.audio_file.file_path
        ]
        print(f"Воспроизведение: {seg.start_time:.2f}s -> {seg.end_time:.2f}s")
        # stdout/stderr уводим в DEVNULL, чтобы не засорять консоль
        subprocess.run(cmd, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL, check = True)

    def clear_screen(self):
        """Очищает экран консоли (Linux/Windows)."""
        if platform.system() == "Windows":
            os.system("cls")
        else:
            os.system("clear")

    def print_status(self):
        """Выводит информацию о текущем состоянии."""
        # self.clear_screen()
        print(f"\n=== Сегмент {self.index + 1} из {len(self.segments)} ===")
        seg = self.current
        print(f"ID: {seg.id} Speaker.id: {seg.speaker.id} AudioFileMarkup.id: {seg.audio_file.id}")
        print(
            f"Таймкоды: {seg.start_time:.2f}s -> {seg.end_time:.2f}s "
            f"[Длительность: {seg.end_time - seg.start_time:.2f}s]"
        )
        print(f"Текст: {seg.text}")
        print(
            "Команды: [p]lay | [n]ext | [b]ack | [g]oto ID | [l]eft+/- dist | [r]ight+/- dist | "
            "[s]plit | [m]erge L/R | [i]mport | [w]rite | [q]uit"
        )

    def run(self):
        """Главный цикл интерфейса."""
        while True:
            self.print_status()

            choice = input("Введите команду: ").strip().lower().split()
            if not choice:
                continue

            cmd = choice[0]
            args = choice[1:]

            if cmd == 'q':
                print("Выход из редактора.")
                break
            if cmd == 'p':
                self.play_segment_sounddevice()
            elif cmd == 'n':
                if self.index < len(self.segments) - 1:
                    self.index += 1
                else:
                    print("Это последний сегмент.")
            elif cmd == 'b':
                if self.index > 0:
                    self.index -= 1
                else:
                    print("Это первый сегмент.")
            elif cmd == 'g':
                if args:
                    self._goto_id(args[0])
                else:
                    print("Укажите ID сегмента.")
            elif cmd in ('l+', 'l-'):
                if args:
                    distance = int(args[0])
                else:
                    distance = 1
                self._shift_boundary('start', distance if cmd == 'l+' else -distance)
            elif cmd in ('r+', 'r-'):
                if args:
                    distance = int(args[0])
                else:
                    distance = 1
                self._shift_boundary('end', distance if cmd == 'r+' else -distance)
            elif cmd == 's':
                self._split_segment()
            elif cmd == 'm':
                if args and args[0] in ('l', 'r'):
                    self._merge_segment(args[0])
                else:
                    print("Укажите направление слияния: m l (слева) или m r (справа)")
            elif cmd == 'i':
                self.load_yaml()
            elif cmd == 'w':
                print("Внимание: вы запросили запись файла")
                confirm = input("Для подтверждения записи введите \"y\": ").strip().lower()
                if confirm == "y":
                    export_markup_to_yaml(
                        yaml_path = self.yaml_path,
                        speakers = self.speakers,
                        audio_file = self.audio_file
                    )
                    print(f"Данные сохранены в YAML файл: {self.yaml_path}")
                else:
                    print("Сохранение данных в YAML файл отменено")
            else:
                print("Неизвестная команда.")

    def _goto_id(self, target_id: str):
        for i, seg in enumerate(self.segments):
            if str(seg.id) == target_id:
                self.index = i
                return
        print(f"Сегмент с ID {target_id} не найден.")

    def _shift_boundary(self, boundary: str, direction: int):
        delta = self.step * direction
        seg = self.current
        if boundary == 'start':
            new_val = max(0.0, seg.start_time + delta)
            if new_val < seg.end_time:
                seg.start_time = round(new_val, 2)
            else:
                print("Левая граница не может быть правее правой.")
        else:
            new_val = seg.end_time + delta
            if new_val > seg.start_time:
                seg.end_time = round(new_val, 2)
            else:
                print("Правая граница не может быть левее левой.")

    def _split_segment(self):
        seg = self.current
        midpoint = round((seg.start_time + seg.end_time) / 2, 2)

        # Создаем второй сегмент (копируем метаданные)
        new_seg = AudioSegmentMarkup(
            id = seg.id + 10000,
            audio_file_id = seg.audio_file_id,
            audio_file = seg.audio_file,
            speaker_id = seg.speaker_id,
            speaker = seg.speaker,
            start_time = midpoint,
            end_time = seg.end_time,
            text = seg.text
        )
        # Обрезаем текущий
        seg.end_time = midpoint

        self.segments.insert(self.index + 1, new_seg)
        print(f"Сегмент разделен пополам на отметке {midpoint}s.")

    def _merge_segment(self, direction: str):
        if direction == 'l':
            if self.index == 0:
                print("Нет сегмента слева для объединения.")
                return
            target_idx = self.index - 1
            left = self.segments[target_idx]
            right = self.current
        else:
            if self.index == len(self.segments) - 1:
                print("Нет сегмента справа для объединения.")
                return
            target_idx = self.index + 1
            left = self.current
            right = self.segments[target_idx]

        # Объединяем таймкоды и текст
        left.end_time = right.end_time
        left.text = f"{left.text or ''} {right.text or ''}".strip()

        # Удаляем поглощенный сегмент
        self.segments.remove(right)

        # Корректируем текущий индекс
        if direction == 'l':
            self.index -= 1
        print("Сегменты успешно объединены.")

def main():
    """Точка входа"""
    args = args_utils.parse_args()
    yaml_path = args.input
    if not yaml_path:
        raise ValueError("Не указан путь к YAML файлу с разметкой")

    editor = AudioSegmentEditor(yaml_path = yaml_path)
    editor.run()

if __name__ == "__main__":
    main()
