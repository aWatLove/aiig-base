"""Добавляет src в sys.path для работы импортов."""

import sys
from pathlib import Path

# Добавляем src в путь для импортов
_src_path = Path(__file__).parent / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

