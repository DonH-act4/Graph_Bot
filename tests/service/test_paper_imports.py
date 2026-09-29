"""Regression test for keeping Docling libraries out of PDF API startup."""

import os
import subprocess
import sys
from pathlib import Path


def test_paper_api_import_is_lightweight():
    env = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import service.papers; "
                "assert 'docling' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
