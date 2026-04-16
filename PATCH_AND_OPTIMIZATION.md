# Инструкция по оптимизации и исправлению Knowledge-RAG MCP Server

В этом документе описаны изменения, внесенные в исходный код `knowledge-rag` для обеспечения стабильности протокола MCP, поддержки специализированных форматов (MQL4/MQL5/Jupyter) и реализации надежной системы исключений.

---

## 1. Исправление стабильности MCP (stdout protection)
**Файл:** `mcp_server/__init__.py`

MCP-протокол использует `stdout` для передачи JSON-сообщений. Если любая библиотека или часть кода вызовет `print()`, это приведет к ошибке «Failed to connect».

**Решение:** Глобальное перенаправление всех вызовов `print()` в `stderr`.

```python
import sys
import builtins

_orig_print = builtins.print
def _stderr_print(*args, **kwargs):
    kwargs.setdefault('file', sys.stderr)
    _orig_print(*args, **kwargs)
builtins.print = _stderr_print
```

---

## 2. Поддержка MetaTrader и Jupyter
**Файл:** `mcp_server/ingestion.py`

Добавлена поддержка расширений `.mqh`, `.mq4` (как текст) и `.ipynb` (как JSON) в таблицу парсеров.

---

## 3. Робастная система исключений (exclude_patterns)
**Файл:** `mcp_server/ingestion.py` и `mcp_server/config.py`

Реализована глубокая проверка компонентов пути для надежного исключения папок типа `.venv`, `.git` или `MQL5`, независимо от их уровня вложенности.

### Изменение в `config.py`:
Исправлена функция `_get_top`, чтобы она корректно загружала списки (List) из YAML-конфига.

### Изменение в `ingestion.py`:
Используется `fnmatch` и разделение пути на части (`Path.parts`) для сопоставления фильтров с реальной структурой директорий.

---

## 4. Защита от лимитов Inotify (Linux)
**Файл:** `mcp_server/server.py`

На Linux-системах сервер теперь не падает при достижении лимита `fs.inotify.max_user_instances`. 

**Решение:** Инициализация `Observer` обернута в `try-except`. Если лимит исчерпан, сервер выводит предупреждение и продолжает работу в режиме ручной индексации.

---

## 5. Постоянный кеш моделей эмбеддингов
**Файл:** `mcp_server/server.py` и `mcp_server/config.py`

Модели `fastembed` больше не хранятся в `/tmp`, где они могут быть удалены системой.

**Решение:** 
1. Добавлено поле `models_cache_dir` в конфиг (по умолчанию `models_cache/` в папке проекта).
2. Параметр `cache_dir` передается в конструктор `TextEmbedding`.

---

## 6. Оптимальный `config.yaml` для проекта

```yaml
paths:
  documents_dir: "/path/to/your/project"
  data_dir: "./data"
  models_cache_dir: "./models_cache"

documents:
  supported_formats:
    - .md
    - .txt
    - .py
    - .mqh
    - .mq4
    - .ipynb

exclude_patterns:
  - "**/MT/MQL5/**"
  - "**/MT/tester/**"
  - "**/MT/MQL4/Files/**"
  - "**/MT/MQL4/Indicators/**"
  - "**/MT/MQL4/Libraries/**"
  - "**/MT/MQL4/Logs/**"
  - "**/MT/MQL4/Profiles/**"
  - "**/MT/MQL4/Scripts/**"
  - "**/MT/MQL4/Trash/**"
  - "**/.venv/**"
  - "**/.git/**"
  - "**/.**"
```
