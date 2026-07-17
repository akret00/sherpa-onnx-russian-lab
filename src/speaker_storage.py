"""Модуль реалиpует хранилище данных пайплайна"""
from abc import ABC, abstractmethod
import copy
from contextlib import contextmanager
import sqlite3
from typing import Any
from collections.abc import Generator
from pathlib import Path
import numpy
import config
from entities import Speaker, SpeakerEmbedding

def make_default_speaker_name(speaker_id: int) -> str:
    """Конструирует имя спикера по умолчанию"""
    return f"SPK_{speaker_id:03d}"

class BaseRepo(ABC):
    """Абстрактный интерфейс хранилища базовых объектов"""
    @abstractmethod
    def load_speakers(self,
        speaker_ids: list[int] | None = None,
        name: str | None = None
    ) -> list[Speaker]:
        """Загружает список спикеров по фильтрам. Без фильтров возвращает ВСЕХ спикеров."""

    @abstractmethod
    def save_speakers(self, speakers: list[Speaker]) -> None:
        """Сохраняет список спикеров. 
        Для новых (id is None) генерирует автоинкремент и мутирует переданный объект.
        """


class InMemoryRepo(BaseRepo):
    """Репозиторий, имитирующий работу СУБД в оперативной памяти (для бенчмарков/тестов)."""
    def __init__(self) -> None:
        self._speaker_id_counter: int = 1
        self._embedding_id_counter: int = 1

        # Хранилище сущностей, где ключ — это уникальный ID спикера
        self._speakers: dict[int, Speaker] = {}

    def load_speakers(
        self,
        speaker_ids: list[int] | None = None,
        name: str | None = None
    ) -> list[Speaker]:
        """Загружает список спикеров по фильтрам. Без фильтров возвращает ВСЕХ спикеров."""
        matched_speakers: list[Speaker] = []

        for speaker in self._speakers.values():
            # Фильтр по списку ID (если передан)
            if speaker_ids is not None and speaker.id not in speaker_ids:
                continue

            # Фильтр по имени (если передан, ищет точное совпадение)
            if name is not None and speaker.name != name:
                continue

            # Возвращаем deepcopy, чтобы изменения объектов в пайплайне
            # не влияли на состояние "БД" до вызова метода сохранения
            matched_speakers.append(copy.deepcopy(speaker))

        return matched_speakers

    def save_speakers(self, speakers: list[Speaker]) -> None:
        """
        Сохраняет список спикеров. 
        Для новых (id is None) генерирует автоинкремент и мутирует переданный объект.
        """
        for speaker in speakers:
            # 1. Если спикер новый, генерируем для него ID и базовое имя
            if speaker.id is None:
                speaker.id = self._speaker_id_counter
                self._speaker_id_counter += 1

                # Если имя осталось дефолтным, дополняем его сгенерированным ID
                if speaker.name is None:
                    speaker.name = make_default_speaker_name(speaker_id = speaker.id)

            # 2. Обрабатываем эмбеддинги спикера
            for emb in speaker.embeddings:
                # Проставляем связь эмбеддинга со спикером
                emb.speaker_id = speaker.id

                # Если у самого эмбеддинга нет ID, выдаем автоинкремент
                if emb.id is None:
                    emb.id = self._embedding_id_counter
                    self._embedding_id_counter += 1

            # 3. Сохраняем глубокую копию состояния объекта в наше хранилище
            self._speakers[speaker.id] = copy.deepcopy(speaker)


class SqliteRepo(BaseRepo):
    """Управляет жизненным циклом БД SQLite и операциями чтения/записи объектов."""
    def __init__(self, db_path: str | None = None) -> None:
        """Инициализирует подключение к БД и создает таблицы, если их нет."""
        if db_path is None:
            # Используем путь из конфига, если ничего не передано
            self.db_path: Path = Path(config.DB_DEFAULT_PATH).expanduser().resolve()
        elif db_path == ":memory:":
            raise ValueError("Работа с БД в памяти не поддерживается в этом репозитории")
        else:
            self.db_path = Path(db_path).expanduser().resolve()

        # Проверяем, нужна ли инициализация базы данных до подключения
        is_need_init = not self.db_path.is_file()
        if is_need_init: # Создаем нужные папки для файла с БД, если их нет
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if is_need_init:
            print("База данных не найдена. Запуск инициализации структуры...")
            self._init_db()
            print(f"Создана база данных: {self.db_path}")

    @contextmanager
    def connection_scope(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Контекстный менеджер управляет транзакциями.
        Закрывает коннект для БД на диске, выполняет автокоммит или автооткат.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Позволяет обращаться к колонке по имени
        conn.execute("PRAGMA foreign_keys = ON;")

        try:
            yield conn
            conn.commit()  # Авто-коммит при успешном завершении блока with
        except Exception:
            conn.rollback()  # Авто-откат при ошибке внутри блока with
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Создает структуру таблиц и индексов, если их нет в базе."""
        # Используем путь к схеме из вашего конфига
        scheme_path = Path(config.DB_DEFAULT_SCHEME_PATH)
        if not scheme_path.is_file():
            raise FileNotFoundError(
                f"Файл со структурой базы данных не найден по пути: {scheme_path}"
            )

        # Читаем весь файл в строку (utf-8 защитит от проблем с кодировкой)
        sql_script = scheme_path.read_text(encoding="utf-8")

        with self.connection_scope() as conn:
            cursor = conn.cursor()
            # executescript() выполняет сразу несколько SQL-команд, разделенных точкой с запятой
            cursor.executescript(sql_script)


    # --- Хелперы для работы с векторами ---
    def _encode_embedding(self, emb: numpy.ndarray | None) -> bytes:
        """Преобразует numpy массив в компактные байты float32."""
        if emb is None:
            return b""
        return emb.astype(numpy.float32).tobytes()

    def _decode_embedding(self, blob: bytes) -> numpy.ndarray:
        """Восстанавливает numpy массив из байтов."""
        return numpy.frombuffer(blob, dtype = numpy.float32)


    # --- Реализация методов интерфейса BaseRepo ---
    def load_speakers(
        self,
        speaker_ids: list[int] | None = None,
        name: str | None = None
    ) -> list[Speaker]:
        """Загружает список спикеров по фильтрам. Без фильтров возвращает ВСЕХ спикеров."""

        # 1. Формируем SQL-запрос для спикеров с динамическими фильтрами
        query_parts = ["SELECT id, name, total_count, created_at FROM speaker WHERE 1=1"]
        params: dict[str, Any] = {}

        if speaker_ids is not None:
            # SQLite не поддерживает списки в именованных параметрах напрямую,
            # генерируем плейсхолдеры вида :id0, :id1...
            id_placeholders = []
            for idx, s_id in enumerate(speaker_ids):
                param_name = f"id_{idx}"
                id_placeholders.append(f":{param_name}")
                params[param_name] = s_id
            query_parts.append(f"AND id IN ({', '.join(id_placeholders)})")

        if name is not None:
            query_parts.append("AND name = :name")
            params["name"] = name

        speakers_query = " ".join(query_parts)
        speakers_dict: dict[int, Speaker] = {}

        with self.connection_scope() as conn:
            cursor = conn.cursor()

            # Шаг 1: Загружаем базовые карточки спикеров
            cursor.execute(speakers_query, params)
            rows = cursor.fetchall()

            if not rows:
                return []

            for row in rows:
                s_id = row["id"]
                speakers_dict[s_id] = Speaker(
                    id = s_id,
                    name = row["name"],
                    total_count = row["total_count"],
                    created_at = row["created_at"],
                    embeddings = []
                )

            # Шаг 2: Загружаем все эмбеддинги для этих спикеров
            # Делаем это одним запросом через IN, чтобы не спамить базу в цикле
            emb_placeholders = [f":s_id_{idx}" for idx in range(len(speakers_dict))]
            emb_params = {f"s_id_{idx}": s_id for idx, s_id in enumerate(speakers_dict.keys())}

            emb_query = f"""
                SELECT id, speaker_id, model_name, embedding 
                FROM speaker_embedding 
                WHERE speaker_id IN ({', '.join(emb_placeholders)})
            """

            cursor.execute(emb_query, emb_params)
            emb_rows = cursor.fetchall()

            for emb_row in emb_rows:
                spk_id = emb_row["speaker_id"]
                vector = self._decode_embedding(emb_row["embedding"])

                embedding_obj = SpeakerEmbedding(
                    id=emb_row["id"],
                    speaker_id=spk_id,
                    model_name=emb_row["model_name"],
                    embedding=vector
                )
                speakers_dict[spk_id].embeddings.append(embedding_obj)

        return list(speakers_dict.values())

    def save_speakers(self, speakers: list[Speaker]) -> None:
        """
        Сохраняет список спикеров.
        Для новых (id is None) генерирует автоинкремент и мутирует переданный объект.
        """
        with self.connection_scope() as conn:
            cursor = conn.cursor()

            for speaker in speakers:
                if speaker.id is None:
                    # Создаем нового спикера
                    cursor.execute(
                        """
                        INSERT INTO speaker (name, total_count, created_at) 
                        VALUES (:name, :total_count, :created_at)
                        """,
                        {
                            "name": speaker.name,
                            "total_count": speaker.total_count,
                            "created_at": speaker.created_at
                        }
                    )
                    speaker.id = cursor.lastrowid

                    # Проверяем дефолтное имя: если оно не изменилось, дополняем полученным ID
                    if speaker.name is None and speaker.id is not None:
                        speaker.name = make_default_speaker_name(speaker_id = speaker.id)
                        cursor.execute(
                            "UPDATE speaker SET name = :name WHERE id = :id",
                            {"name": speaker.name, "id": speaker.id}
                        )
                else:
                    # Обновляем существующего спикера (честный UPSERT/Update)
                    cursor.execute(
                        """
                        UPDATE speaker 
                        SET name = :name, total_count = :total_count 
                        WHERE id = :id
                        """,
                        {"name": speaker.name, "total_count": speaker.total_count, "id": speaker.id}
                    )

                # Сохраняем эмбеддинги текущего спикера
                for emb in speaker.embeddings:
                    emb.speaker_id = speaker.id
                    blob_data = self._encode_embedding(emb.embedding)

                    if emb.id is None:
                        # Новый эмбеддинг для этой модели
                        cursor.execute(
                            """
                            INSERT INTO speaker_embedding (speaker_id, model_name, embedding) 
                            VALUES (:speaker_id, :model_name, :embedding)
                            """,
                            {
                                "speaker_id": emb.speaker_id,
                                "model_name": emb.model_name,
                                "embedding": blob_data
                            }
                        )
                        emb.id = cursor.lastrowid
                    else:
                        # Обновляем существующий вектор (например, если модель уточнила эмбеддинг)
                        cursor.execute(
                            """
                            UPDATE speaker_embedding 
                            SET embedding = :embedding 
                            WHERE id = :id
                            """,
                            {"embedding": blob_data, "id": emb.id}
                        )

def create_spk_repo(pl_conf: config.PipelineConfig) -> BaseRepo:
    """Создает репозиторий спикеров на основе типа репо в конфигурации пайплайна"""
    if pl_conf.diar_vad.speaker_repo_type is config.SpeakerRepoType.IN_MEMORY:
        return InMemoryRepo()
    if pl_conf.diar_vad.speaker_repo_type is config.SpeakerRepoType.DB_SQLITE:
        return SqliteRepo(db_path = pl_conf.diar_vad.db_path)
    raise ValueError(f"Недопустимый тип репозитория: {pl_conf.diar_vad.speaker_repo_type}")
