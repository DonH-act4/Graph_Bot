import os
from pathlib import Path


def test_mock_env_preserves_home_and_clears_app_credentials(mock_env) -> None:
    assert Path.home().is_dir()
    assert "OPENAI_API_KEY" not in os.environ
