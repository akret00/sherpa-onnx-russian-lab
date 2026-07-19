-- Включаем поддержку внешних ключей (рекомендуется для SQLite)
-- Закомментирована потому, что включается при создании коннекта в коде
-- PRAGMA foreign_keys = ON;

-- Таблица спикеров
CREATE TABLE IF NOT EXISTS speaker (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT DEFAULT 'Unknown Speaker',
    total_count INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Таблица эмбеддингов голосов
CREATE TABLE IF NOT EXISTS speaker_embedding (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (speaker_id) REFERENCES speaker(id) ON DELETE CASCADE
);

-- Таблица аудиофайлов
CREATE TABLE IF NOT EXISTS audio_file (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL UNIQUE,
    duration_seconds REAL NOT NULL,
    processed_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Таблица сегментов речи
CREATE TABLE IF NOT EXISTS speech_segment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_file_id INTEGER NOT NULL,
    speaker_id INTEGER NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    text TEXT,
    word_count INTEGER,
    FOREIGN KEY (audio_file_id) REFERENCES audio_file(id) ON DELETE CASCADE,
    FOREIGN KEY (speaker_id) REFERENCES speaker(id) ON DELETE CASCADE
);

-- Создаем индексы для ускорения выборок аналитики и фрагментов
CREATE INDEX IF NOT EXISTS idx_speaker_name ON speaker(name);
CREATE INDEX IF NOT EXISTS idx_embedding_model ON speaker_embedding(model_name);
CREATE INDEX IF NOT EXISTS idx_embedding_speaker ON speaker_embedding(speaker_id);
CREATE INDEX IF NOT EXISTS idx_seg_speaker ON speech_segment(speaker_id);
CREATE INDEX IF NOT EXISTS idx_seg_audio ON speech_segment(audio_file_id);
