"""Модуль реалиpует хранилище данных о спикерах"""
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
import sqlite3
from typing import List
from pathlib import Path
import numpy
import config

# ==========================================
# 1. КЛАССЫ ДАННЫХ (ОБЪЕКТНЫЕ МОДЕЛИ)
# ==========================================

class SpeakerUpdateMode(Enum):
    """Режимы обновления существующих спикеров при сохранении."""
    UPDATE_ALL_EXCEPT_EMBEDDING = "all_except_embedding" # Обновить имя, метрики (если появятся)
    UPDATE_EMBEDDINGS_ONLY = "embeddings_only"           # Обновить только вектор
    NO_UPDATE = "no_update"                              # Не трогать старых, только добавлять новых

@dataclass
class Speaker:
    """Модель данных спикера."""
    id: int | None = None
    name: str = "Unknown Speaker"
    embedding: numpy.ndarray | None = None
    total_count: int = 0 # Глобальный счетчик фраз
    count: int = 0  # Сессионный счетчик фраз, не сохраняется в БД
    created_at: str | None = None

@dataclass
class AudioFile:
    """Модель метаданных аудиофайла."""
    id: int | None = None
    file_path: str = ""
    duration_seconds: float = 0.0
    processed_at: str | None = None

@dataclass
class AudioSegment:
    """Модель текстового сегмента аудиофайла с таймкодами."""
    id: int | None = None
    audio_file_id: int | None = None
    speaker_id: int | None = None
    start_time: float = 0.0
    end_time: float = 0.0
    text: str | None = None
    word_count: int | None = None


# ==========================================
# 2. ОСНОВНОЙ РЕПОЗИТОРИЙ БАЗЫ ДАННЫХ
# ==========================================

class VoiceDbRepository:
    """Управляет жизненным циклом БД SQLite и операциями чтения/записи объектов."""

    def __init__(self, db_path: str | None = None):
        """Инициализирует подключение к БД и создает таблицы, если их нет."""
        # Корректно разворачиваем пути (~, относительные и т.д.) в абсолютный путь
        if db_path is None:
            self.db_path = config.DB_DEFAULT_PATH
        elif db_path == ":memory:":
            raise ValueError("Работа с БД в памяти не поддерживается")
        else:
            self.db_path = Path(db_path).expanduser().resolve()

        # Проверяем, нужна ли инициализация базы данных
        is_need_init = False
        # Проверяем, существует ли файл базы данных ДО подключения
        is_need_init = not self.db_path.is_file()
        if is_need_init:
            # Создаем родительские папки, если их еще нет
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Инициализируем структуру БД, если она новая
        if is_need_init:
            print("База данных не найдена. Запуск инициализации структуры...")
            self._init_db()
            print(f"Создана база данных: {self.db_path}")

    @contextmanager
    def connection_scope(self):
        """
        Контекстный менеджер управляет транзакциями.
        Закрывает коннект для БД на диске, или оставляет открытым для БД в ОЗУ.
        """
        # 1. Получаем нужный коннект
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row # Позволяет обращаться к колонке по имени
        conn.execute("PRAGMA foreign_keys = ON;")

        try:
            yield conn
            conn.commit()  # Авто-коммит при успешном завершении блока with
        except Exception:
            conn.rollback()  # Авто-откат при ошибке внутри блока with
            raise
        finally:
            # 2. Закрываем коннект
            conn.close()

    def _init_db(self):
        """Создает структуру таблиц и индексов, которых нет в базе."""
        with self.connection_scope() as conn:
            cursor = conn.cursor()

            # Проверяем, существует ли SQL-файл со схемой БД
            if not config.DB_DEFAULT_SCHEME_PATH.is_file():
                raise FileNotFoundError(
                    f"Файл со структурой базы данных не найден по пути: "
                    f"{config.DB_DEFAULT_SCHEME_PATH}"
                )

            # Читаем весь файл в строку (utf-8 защитит от проблем с кодировкой)
            sql_script = config.DB_DEFAULT_SCHEME_PATH.read_text(encoding="utf-8")

            # executescript() выполняет сразу несколько SQL-команд, разделенных точкой с запятой
            cursor.executescript(sql_script)

    # --- МЕТОДЫ ЗАГРУЗКИ (READ) ---

    def load_speakers(self,
        speaker_ids: List[int] | None = None,
        name: str | None = None
    ) -> List[Speaker]:
        """Загружает список спикеров по фильтрам. Без фильтров возвращает ВСЕХ спикеров."""
        query = "SELECT id, name, embedding_blob, total_count, created_at FROM speaker WHERE 1=1"
        params = []

        if speaker_ids:
            placeholders = ",".join(["?"] * len(speaker_ids))
            query += f" AND id IN ({placeholders})"
            params.extend(speaker_ids)

        if name:
            query += " AND name LIKE ?"
            params.append(f"%{name}%")

        with self.connection_scope() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [
            Speaker(
                id = r["id"],
                name = r["name"],
                embedding = numpy.frombuffer(r["embedding_blob"], dtype = numpy.float32),
                total_count = r["total_count"],
                created_at = r["created_at"],
            )
            for r in rows
        ]

    def load_audio_file(
        self,
        file_id: int | None = None,
        file_path: str | None = None
    ) -> AudioFile | None:
        """Ищет аудиофайл в базе по его ID или по уникальному системному пути."""
        query = "SELECT id, file_path, duration_seconds, processed_at FROM audio_file WHERE "
        if file_id is not None:
            query += "id = ?"
            param = file_id
        elif file_path is not None:
            query += "file_path = ?"
            param = file_path
        else:
            return None

        with self.connection_scope() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (param,))
            row = cursor.fetchone()

        if row:
            return AudioFile(
                id=row["id"],
                file_path=row["file_path"],
                duration_seconds=row["duration_seconds"],
                processed_at=row["processed_at"]
            )
        return None

    def load_file_content(self, audio_file_id: int):
        """
        Возвращает кортеж: (Список объектов AudioSegment, Список объектов Speaker),
        задействованных в файле.
        """
        segments: List[AudioSegment] = []

        # 1. Загружаем все сегменты для файла
        with self.connection_scope() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, audio_file_id, speaker_id, start_time, end_time, text, word_count
                FROM speech_segment
                WHERE audio_file_id = ?
                ORDER BY start_time ASC
            """, (audio_file_id,))
            seg_rows = cursor.fetchall()

        for r in seg_rows:
            segments.append(AudioSegment(
                id=r["id"],
                audio_file_id=r["audio_file_id"],
                speaker_id=r["speaker_id"],
                start_time=r["start_time"],
                end_time=r["end_time"],
                text=r["text"],
                word_count=r["word_count"]
            ))

        # 2. Извлекаем уникальные ID спикеров из этих сегментов и загружаем их
        unique_speaker_ids = list(
            {seg.speaker_id for seg in segments if seg.speaker_id is not None}
        )
        speakers = self.load_speakers(speaker_ids=unique_speaker_ids) if unique_speaker_ids else []

        return segments, speakers

    # --- МЕТОДЫ СОХРАНЕНИЯ (WRITE) ---

    def save_speakers(
        self,
        speakers: List[Speaker],
        mode: SpeakerUpdateMode = SpeakerUpdateMode.NO_UPDATE
    ):
        """Сохраняет список спикеров. 
        
        Для новых (id is None) генерирует автоинкремент и мутирует переданный объект.
        Для старых применяет один из выбранных режимов обновления `mode`.
        """
        with self.connection_scope() as conn:
            cursor = conn.cursor()

            for speaker in speakers:
                # Сценарий А: Абсолютно новый спикер
                if speaker.id is None:
                    cursor.execute(
                        "INSERT INTO speaker (name, embedding_blob, total_count) "
                        "VALUES (:name, :embedding_blob, :total_count);",
                        {
                            "name": speaker.name,
                            "embedding_blob": speaker.embedding.tobytes(),
                            "total_count": speaker.total_count,
                        }
                    )
                    speaker.id = cursor.lastrowid  # Наполняем объект сгенерированным ID из базы

                # Сценарий Б: Спикер уже существует, обрабатываем согласно логике mode
                else:
                    if mode == SpeakerUpdateMode.UPDATE_ALL_EXCEPT_EMBEDDING:
                        cursor.execute(
                            "UPDATE speaker "
                            "SET name = :name, total_count = :total_count "
                            "WHERE id = :id;",
                            {
                                "name": speaker.name,
                                "total_count": speaker.total_count,
                                "id": speaker.id
                            }
                        )
                    elif mode == SpeakerUpdateMode.UPDATE_EMBEDDINGS_ONLY:
                        cursor.execute(
                            "UPDATE speaker "
                            "SET embedding_blob = :embedding_blob "
                            "WHERE id = :id;",
                            {
                                "embedding_blob": speaker.embedding.tobytes(),
                                "id": speaker.id
                            }
                        )
                    elif mode == SpeakerUpdateMode.NO_UPDATE:
                        pass  # Ничего не делаем со старым спикером

    def save_audio_file(self, audio_file: AudioFile) -> int:
        """Сохраняет карточку аудиофайла. Возвращает ID записи (включая мутацию объекта)."""
        with self.connection_scope() as conn:
            cursor = conn.cursor()
            if audio_file.id is None:
                cursor.execute(
                    "INSERT INTO audio_file (file_path, duration_seconds) "
                    "VALUES (:file_path, :duration_seconds);",
                    {
                        "file_path": audio_file.file_path,
                        "duration_seconds": audio_file.duration_seconds
                    }
                )
                audio_file.id = cursor.lastrowid
            else:
                cursor.execute(
                    "UPDATE audio_file "
                    "SET file_path = :file_path, duration_seconds = :duration_seconds "
                    "WHERE id = :id;",
                    {
                        "file_path": audio_file.file_path,
                        "duration_seconds": audio_file.duration_seconds,
                        "id": audio_file.id
                    }
                )

        return audio_file.id

    def save_audio_segments(self, audio_file_id: int, segments: List[AudioSegment]):
        """
        Сохраняет пачку сегментов (таймкодов) для конкретного аудиофайла в рамках
        единой транзакции.
        """
        # Подготавливаем кортежи для быстрой пакетной вставки через executemany
        data_to_insert = [
            {
                "audio_file_id": audio_file_id,
                "speaker_id": seg.speaker_id,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "text": seg.text,
                "word_count": seg.word_count
            }
            for seg in segments
        ]

        with self.connection_scope() as conn:
            cursor = conn.cursor()

            # Перед перезаписью сегментов конкретного файла удаляем старые,
            # если требуется логикой перезаписи
            cursor.execute(
                "DELETE FROM speech_segment WHERE audio_file_id = :audio_file_id;",
                {"audio_file_id": audio_file_id,}
            )

            cursor.executemany("""
                INSERT INTO speech_segment
                (audio_file_id, speaker_id, start_time, end_time, text, word_count)
                VALUES (:audio_file_id, :speaker_id, :start_time, :end_time, :text, :word_count);
            """, data_to_insert)

def usage_sample():
    """Пример использования функционала класса"""
    # 1. Инициализируем репозиторий (создаст файл 'speech_vault.db', если его нет)
    repo = VoiceDbRepository()

    # 2. Создаем тестовых спикеров (один старый, один новый рантайм-спикер)
    speaker_existing = Speaker(id=1, name="John Doe Modified", embedding=b'\x01\x02\x03')
    speaker_new_found = Speaker(name="SPEAKER_01 (Unknown)", embedding=b'\x09\x09\x09')

    speakers_list = [speaker_existing, speaker_new_found]

    # 3. Сохраняем спикеров с обновлением имени для старых.
    # Обратите внимание: у speaker_new_found динамически заполнится поле .id!
    repo.save_speakers(speakers_list, mode=SpeakerUpdateMode.UPDATE_ALL_EXCEPT_EMBEDDING)
    print(f"Новому спикеру база присвоила ID: {speaker_new_found.id}")

    # 4. Создаем и сохраняем информацию об обработанном аудиофайле
    audio = AudioFile(file_path="/raw/audio/interview_05.wav", duration_seconds=124.5)
    file_id = repo.save_audio_file(audio)

    # 5. Генерируем пачку сегментов диаризации, привязывая их к выданным базой ID спикеров
    runtime_segments = [
        AudioSegment(
            speaker_id=speaker_existing.id,
            start_time=0.0,
            end_time=15.2,
            text="Привет всем",
            word_count=2
        ),
        AudioSegment(
            speaker_id=speaker_new_found.id,
            start_time=15.2,
            end_time=30.0,
            text="Здравствуйте!",
            word_count=1
        )
    ]
    repo.save_audio_segments(audio_file_id=file_id, segments=runtime_segments)

    # ==========================================
    # ПРОВЕРКА ЧТЕНИЯ (Где-то в интерфейсе разметчика)
    # ==========================================

    # Загружаем контент файла обратно
    loaded_segments, loaded_speakers = repo.load_file_content(audio_file_id=file_id)

    print(f"\nВ файле обнаружено сегментов: {len(loaded_segments)}")
    print(f"\nВ файле обнаружено спикеров: {len(loaded_speakers)}")
    for spk in loaded_speakers:
        print(f"- Спикер ID {spk.id}: {spk.name}")
