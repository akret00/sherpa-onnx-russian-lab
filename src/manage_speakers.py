"""
Менеджер голосовых профилей (CLI)
Управление спикерами: просмотр, переименование, сравнение, удаление
"""
import os
import platform
import math
from enum import Enum, auto
from abc import ABC, abstractmethod
import numpy
from entities import Speaker, SpeakerEmbedding
from speaker_storage import SqliteRepo

# --------------------------------------------------------------
# Конфигурация
# --------------------------------------------------------------
MAX_DATA_ROWS = 10
FORM_COLS = 2
PAGE_SIZE = MAX_DATA_ROWS * FORM_COLS # Сколько спикеров на одной странице
COL_WIDTH_ID = 3
COL_WIDTH_NAME = 15
COL_WIDTH_SEGMENTS = 8
COL_WIDTH_VECTOR = 8
COLS_DELIMITER = " | "
FORM_WIDTH = (
    (COL_WIDTH_ID + COL_WIDTH_NAME + COL_WIDTH_SEGMENTS + COL_WIDTH_VECTOR +3) * FORM_COLS
    + len(COLS_DELIMITER) * (FORM_COLS - 1)
)
FILL_MODE_HORISINTAL = "horisontal"
FILL_MODE_VERTICAL = "vertical"
FILL_MODE = FILL_MODE_VERTICAL
# FILL_MODE = FILL_MODE_HORISINTAL

class FormType(Enum):
    """Содержит типы форм"""
    SPEAKER_LIST = auto()
    SPEAKER = auto()

class AppAction(Enum):
    """Содержит действия приложения"""
    EXIT = auto()
    OPEN_DETAIL = auto()
    BACK = auto()
    REFRESH = auto()

# --------------------------------------------------------------
# Вспомогательные функции
# --------------------------------------------------------------
def cosine_similarity(emb1: numpy.ndarray, emb2: numpy.ndarray) -> float:
    """Вычисляет косинусное сходство между двумя эмбеддингами."""
    if emb1 is None or emb2 is None:
        return 0.0
    return float(numpy.dot(emb1, emb2)) # Косинусное для нормированных векторов

def press_enter_to_continue() -> None:
    """Ждет нажатия Enter."""
    input("\nНажмите Enter для продолжения...")

def get_int_input(prompt: str, min_val: int = 1, max_val: int | None = None) -> int | None:
    """
    Безопасный ввод целого числа.
    Возвращает None, если ввод пустой (отмена).
    """
    while True:
        raw = input(prompt).strip()
        if not raw:
            return None
        try:
            val = int(raw)
            if val < min_val or (max_val is not None and val > max_val):
                print(f"Пожалуйста, введите число от {min_val}"
                      + (f" до {max_val}" if max_val else ""))
                continue
            return val
        except ValueError:
            print("Пожалуйста, введите целое число.")


# --------------------------------------------------------------
# Операции со спикерами
# --------------------------------------------------------------

def rename_speaker(repo: SqliteRepo) -> bool:
    """Переименование спикера. Возвращает True, если что-то изменилось."""
    sp_id = get_int_input("Введите ID спикера для переименования (Enter для отмены): ")
    if sp_id is None:
        return False

    # Загружаем одного спикера
    speakers = repo.load_speakers(speaker_ids=[sp_id])
    if not speakers:
        print(f"Спикер с ID={sp_id} не найден.")
        return False

    sp = speakers[0]
    print(f"\nТекущее имя: {sp.name}")
    new_name = input("Новое имя: ").strip()

    if not new_name:
        print("Переименование отменено (пустое имя).")
        return False

    if new_name == sp.name:
        print("Имя не изменилось.")
        return False

    # Сохраняем с новым именем
    sp.name = new_name
    repo.save_speakers(speakers = [sp])
    print(f"Спикер ID={sp.id} переименован в '{new_name}'")
    return True


def compare_speakers(repo: SqliteRepo) -> None:
    """Сравнение двух спикеров по косинусному сходству эмбеддингов."""
    print("\nСравнение двух спикеров по сходству голосов:")

    id1 = get_int_input("Введите ID первого спикера (Enter для отмены): ")
    if id1 is None:
        return

    id2 = get_int_input("Введите ID второго спикера (Enter для отмены): ")
    if id2 is None:
        return

    if id1 == id2:
        print("Вы выбрали одного и того же спикера. Сходство всегда 100%.")
        return

    # Загружаем обоих
    speakers = repo.load_speakers(speaker_ids=[id1, id2])
    sp_dict = {sp.id: sp for sp in speakers}

    if id1 not in sp_dict:
        print(f"Спикер с ID={id1} не найден.")
        return
    if id2 not in sp_dict:
        print(f"Спикер с ID={id2} не найден.")
        return

    sp1, sp2 = sp_dict[id1], sp_dict[id2]

    # Проверяем наличие эмбеддингов
    if sp1.embeddings is None or sp2.embeddings is None:
        print("У одного из спикеров отсутствуют эмбеддинги (голосовые образцы).")
        return

    # Создаем словарь для имени модели и эмбеддингов
    table: dict[str, list[SpeakerEmbedding | None | float]] = {}
    # Заполняем словарь данными
    for emb in sp1.embeddings:
        table.setdefault(emb.model_name, [None, None, None])
        table[emb.model_name][0] = emb
    for emb in sp2.embeddings:
        table.setdefault(emb.model_name, [None, None, None])
        table[emb.model_name][1] = emb
        emb1, emb2, _ = table[emb.model_name]
        if isinstance(emb1, SpeakerEmbedding) and isinstance(emb2, SpeakerEmbedding):
            if emb1.embedding is not None and emb2.embedding is not None:
                table[emb.model_name][2] = cosine_similarity(emb1.embedding, emb2.embedding)

    # Печать заголовка отчета
    print(f"\n{'=' * 55}")
    print(f"Спикер 1: [{sp1.id}] {sp1.name} | Спикер 2: [{sp2.id}] {sp2.name}")

    for model_name, values in table.items():
        score = values[2]
        # Проверяем, что оценка была успешно посчитана и записана
        if isinstance(score, float):
            print(f"Модель: {model_name:<15} | Косинусная близость: {score:.4f}")
        elif values[0] is None:
            print(f"Модель: {model_name:<15} | Нет эмбеддинга для Спикера 1")
        else:
            print(f"Модель: {model_name:<15} | Нет эмбеддинга для Спикера 2")

    print(f"{'=' * 55}")


def delete_speaker(repo: SqliteRepo) -> bool:
    """
    Удаление спикера с обязательным подтверждением.
    Возвращает True, если удаление выполнено.
    """
    sp_id = get_int_input("Введите ID спикера для удаления (Enter для отмены): ")
    if sp_id is None:
        return False

    speakers = repo.load_speakers(speaker_ids=[sp_id])
    if not speakers:
        print(f"Спикер с ID={sp_id} не найден.")
        return False

    sp = speakers[0]

    # Предупреждение, если есть записи
    print("\nВНИМАНИЕ! Вы собираетесь удалить спикера:")
    print(f"   ID: {sp.id}")
    print(f"   Имя: {sp.name}")
    print(f"   Количество фраз: {sp.total_count}")

    # Двойное подтверждение
    confirm = input("\nДля подтверждения удаления введите 'del': ").strip().lower()
    if confirm != "del":
        print("Удаление отменено.")
        return False

    try:
        # Временное решение, пока нет метода delete:
        repo.delete_speaker(sp_id)
        print(f"Спикер ID={sp_id} удален.")
    except RuntimeError as e:
        print(f"Ошибка при удалении спикера: {e}")
        return False

    return True

# --- БАЗОВАЯ ФОРМА ---
class BaseForm(ABC):
    """Базовая форма"""
    def __init__(self, repo: SqliteRepo) -> None:
        self._repo = repo

    def clear_screen(self) -> None:
        """Очищает экран консоли (Linux/Windows)."""
        if platform.system() == "Windows":
            os.system("cls")
        else:
            os.system("clear")

    @abstractmethod
    def render(self) -> None:
        """Отрисовать форму и меню"""

    @abstractmethod
    def handle_command(self, cmd: str) -> AppAction:
        """
        Обработать команду и вернуть строковый статус.
        Например: 'back', 'exit', 'edit_speaker', или 'refresh'.
        """


# --- ФОРМА 1: СПИСОК СПИКЕРОВ ---
class SpeakerListForm(BaseForm):
    """Форма со списком спикеров"""
    def __init__(self, repo: SqliteRepo) -> None:
        super().__init__(repo)
        self.selected_speaker_id = None # Сюда сохраним выбор, если он будет
        self.page = 1   # текущая страница
        self.search_query = ""
        self.total_pages = 1
        self.total_count = 0    # общее количество спикеров (для пагинации)

    def _format_speaker(self, sp: Speaker | None) -> str:
        """Форматирует одного спикера в одну группу колонок"""
        if sp is None:
            # Возвращаем пустые колонки, если данных для этого места нет
            return (
                f" {' ':<{COL_WIDTH_ID}}"
                f" {' ':<{COL_WIDTH_NAME}}"
                f" {' ':<{COL_WIDTH_SEGMENTS}}"
                f" {' ':<{COL_WIDTH_VECTOR}}"
            )

        name = sp.name if sp.name else "Без имени"
        if len(name) > COL_WIDTH_NAME:
            display_name = name[:COL_WIDTH_NAME - 2] + ".."
        else:
            display_name = name

        return (
            f" {sp.id:<{COL_WIDTH_ID}} "
            f"{display_name:<{COL_WIDTH_NAME}} "
            f"{sp.total_count:<{COL_WIDTH_SEGMENTS}}"
            f"{len(sp.embeddings):<{COL_WIDTH_VECTOR}}"
        )

    def _show_header(self) -> None:
        """Показывает заголовок формы"""
        # print(f"\n{'═' * FORM_WIDTH}")
        print("  МЕНЕДЖЕР ГОЛОСОВЫХ ПРОФИЛЕЙ")
        print(f"\n{'═' * FORM_WIDTH}")

    def _show_speakers_table(self, speakers: list[Speaker]) -> None:
        """
        Выводит таблицу спикеров с пагинацией.
        Args:
            speakers: список спикеров для отображения (уже отфильтрованный)
        """
        if not speakers:
            print("\n   Спикеры не найдены.")
            return

        # Заголовок
        # print(f"\n{'-' * FORM_WIDTH}")
        col_title = (
            f" {'ID':<{COL_WIDTH_ID}} "
            f"{'Имя':<{COL_WIDTH_NAME}} "
            f"{'Фразы':<{COL_WIDTH_SEGMENTS}}"
            f"{'Векторы':<{COL_WIDTH_VECTOR}}"
        )
        # Повторяет заголовок по количеству колонок формы FORM_COLS
        print(COLS_DELIMITER.join(col_title for _ in range(FORM_COLS)))
        print(f"{'-' * FORM_WIDTH}")

        # Данные
        if FILL_MODE == FILL_MODE_VERTICAL:
            # --- РЕЖИМ 1: Сверху вниз, затем следующая колонка (По вертикали) ---
            # Рассчитываем, сколько строк реально понадобится
            # (но не больше лимита для заполнения колонок)
            total_needed_rows = math.ceil(len(speakers) / FORM_COLS)
            rows_count = min(MAX_DATA_ROWS, total_needed_rows)

            for row_idx in range(rows_count):
                row_cells: list[Speaker | None] = []
                for col_idx in range(FORM_COLS):
                    # Вычисляем индекс элемента в исходном списке для текущей строки и колонки
                    element_idx = col_idx * rows_count + row_idx

                    if element_idx < len(speakers):
                        row_cells.append(speakers[element_idx])
                    else:
                        row_cells.append(None)

                print(COLS_DELIMITER.join(self._format_speaker(sp) for sp in row_cells))
        else:
            # --- РЕЖИМ 2: По очереди во все колонки (Слева направо, сверху вниз) ---
            for i in range(0, len(speakers), FORM_COLS):
                # Берем срез элементов для одной строки таблицы
                chunk = speakers[i:i + FORM_COLS]
                # Если элементов меньше, чем колонок, дополняем None
                while len(chunk) < FORM_COLS:
                    chunk.append(None)

                # Форматируем каждый элемент и соединяем их в одну строку
                print(COLS_DELIMITER.join(self._format_speaker(sp) for sp in chunk))

        print(f"{'-' * FORM_WIDTH}")

        # Информация о странице
        if self.total_count > PAGE_SIZE:
            total_pages = (self.total_count + PAGE_SIZE - 1) // PAGE_SIZE
            print(f"  Страница {self.page}/{total_pages} (всего спикеров: {self.total_count})")
        else:
            print(f"  Всего спикеров: {len(speakers)}")

    def _show_menu(self) -> None:
        """Показывает главное меню."""
        print(f"{'═' * FORM_WIDTH}")
        print("  [P] Предыдущая страница    [N] Следующая страница")
        print("  [S] Поиск по имени         [A] Показать всех")
        print(f"{'-' * FORM_WIDTH}")
        print("  [I] Инфо о спикере         [C] Сравнить спикеров")
        print("  [R] Переименовать спикера  [M] Объединить спикеров")
        print("  [D] Удалить спикера")
        print(f"{'-' * FORM_WIDTH}")
        print("  [Q] Выход")
        print(f"{'═' * FORM_WIDTH}")

        if self.search_query:
            print(f"  Поиск: '{self.search_query}'")

    def render(self) -> None:
        """Рендеринг формы"""
        super().clear_screen()

        # Загружаем спикеров с учетом поиска
        if self.search_query:
            speakers = self._repo.load_speakers(name = self.search_query)
        else:
            speakers = self._repo.load_speakers()

        self.total_count = len(speakers)
        self.total_pages = max(1, (self.total_count + PAGE_SIZE - 1) // PAGE_SIZE)

        # Корректируем страницу
        self.page = min(max(self.page, 1), self.total_pages)

        # Нарезаем страницу
        start_idx = (self.page - 1) * PAGE_SIZE
        end_idx = start_idx + PAGE_SIZE
        page_speakers = speakers[start_idx:end_idx]

        # Показываем заголовок
        self._show_header()
        # Показываем таблицу
        self._show_speakers_table(page_speakers)
        # Показываем меню
        self._show_menu()

    def handle_command(self, cmd: str) -> AppAction:
        if cmd == 'q':
            print("До свидания!")
            return AppAction.EXIT

        if cmd == 'p':
            if self.page > 1:
                self.page -= 1
        elif cmd == 'n':
            if self.page < self.total_pages:
                self.page += 1
        elif cmd == 's':
            self.search_query = input("Введите имя или часть имени для поиска: ").strip()
            self.page = 1  # Сбрасываем страницу при новом поиске
            if not self.search_query:
                self.search_query = ""  # Показываем всех
        elif cmd == 'a':
            self.search_query = ""
            self.page = 1
        elif cmd == 'r':
            if rename_speaker(self._repo):
                press_enter_to_continue()
        elif cmd == 'c':
            compare_speakers(self._repo)
            press_enter_to_continue()
        elif cmd == 'd':
            if delete_speaker(self._repo):
                press_enter_to_continue()
        elif cmd.isdigit():
            # Прямой переход на страницу
            target_page = int(cmd)
            if 1 <= target_page <= self.total_pages:
                self.page = target_page
            else:
                print(f"Нет страницы {target_page}. Доступны страницы 1-{self.total_pages}")
                press_enter_to_continue()

        else:
            print("Неизвестная команда. Используйте буквы из меню.")
            press_enter_to_continue()

        return AppAction.REFRESH # Если команда не распознана, просто обновим экран


# --- ФОРМА 2: УПРАВЛЕНИЕ ОДНИМ СПИКЕРОМ ---
class SpeakerDetailForm(BaseForm):
    """Форма управления одним спикером"""
    def __init__(self, repo: SqliteRepo) -> None:
        super().__init__(repo)
        self.speaker_id: int

    def render(self) -> None:
        self.clear_screen()
        # Предположим, у repo есть метод get_by_id
        speaker = self.repo.get_by_id(self.speaker_id) 
        print(f"=== УПРАВЛЕНИЕ СПИКЕРОМ: {speaker['name']} ===")
        print("1: Изменить имя")
        print("2: Удалить спикера")
        print("\n[Меню] b: Назад в список")

    def handle_command(self, cmd: str) -> AppAction:
        if cmd.lower() == 'b':
            return 'back'

        if cmd == '1':
            new_name = input("Новое имя: ")
            self.repo.update_name(self.speaker_id, new_name)
            return 'refresh'

        return 'refresh'


class SpeakerEditorApp:
    """Главный класс редактора спикеров"""
    def __init__(self) -> None:
        self._repo = SqliteRepo()
        self.is_running = True
        # Храним состояние: на каком мы этапе и с каким спикером работаем
        self.current_form = FormType.SPEAKER_LIST
        # self.current_speaker_id: int | None = None
        self.spk_list_form = SpeakerListForm(self._repo)
        self.spk_detail_form = SpeakerDetailForm(self._repo)

    def run(self) -> None:
        """Главный цикл приложения."""
        while self.is_running:
            # 1. Определение нужной формы
            form: BaseForm
            if self.current_form is FormType.SPEAKER_LIST:
                form = self.spk_list_form
            else:
                form = self.spk_detail_form
            # 2. Отображение
            form.render()
            # 3. Ввод команды
            cmd = input("\nВаш выбор: ").strip().lower()
            # 4. Обработка команды самой формой
            action = form.handle_command(cmd)
            # 5. Диспетчеризация (EditorApp решает, куда переходить)
            self._route(action, form)

    def _route(self, action: AppAction, form: BaseForm) -> None:
        """Логика переключения между экранами"""
        if action is AppAction.EXIT:
            self.is_running = False
        elif action is AppAction.OPEN_DETAIL:
            # Забираем ID, который выбрал пользователь внутри SpeakerListForm
            self.current_speaker_id = form.selected_speaker_id
            self.current_state = "DETAIL"
        elif action is AppAction.BACK:
            self.current_state = "LIST"
            self.current_speaker_id = None
        elif action is AppAction.REFRESH:
            pass # Ничего не меняем, цикл просто запустится заново


# --------------------------------------------------------------
# Точка входа
# --------------------------------------------------------------
if __name__ == "__main__":
    app = SpeakerEditorApp()
    app.run()
