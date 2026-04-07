from pathlib import Path
import sys


# Make local package imports stable for both `pytest` and `python -m pytest`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
