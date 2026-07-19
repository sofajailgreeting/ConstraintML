import sys
from pathlib import Path

# Allow running tests without an editable install (e.g. before pip install -e completes).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
