import os
import subprocess
import sys
from pathlib import Path


def prepare_moonshot_dataset(
    index_path: str = "~/MoonshotDatasetv3/index.pkl",
    out_dir: str = "~/MoonshotDatasetv3",
    moonshot_root: str | None = None,
) -> None:
    """
    Call SMART-Moonshot's fp_loader so that the MoonshotDatasetv3
    assets are prepared locally.

    We reuse your current Python (the .venv-312 one) and point
    PYTHONPATH at SMART-Moonshot/src so that `modules.data.fp_loader`
    is importable.
    """
    index = Path(index_path).expanduser()
    out_dir_path = Path(out_dir).expanduser()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Default location where you cloned SMART-Moonshot
    if moonshot_root is None:
        moonshot_root = str(Path.home() / "SMART-Moonshot")

    moonshot_root_path = Path(moonshot_root)
    if not moonshot_root_path.exists():
        raise FileNotFoundError(
            f"Could not find SMART-Moonshot repo at {moonshot_root_path}. "
            "Pass moonshot_root explicitly or adjust the default path."
        )

    # Equivalent to:
    #   cd SMART-Moonshot
    #   PYTHONPATH=src python -m modules.data.fp_loader fragments ...
    cmd = [
        sys.executable,
        "-m",
        "modules.data.fp_loader",
        "fragments",
        "--index",
        str(index),
        "--out-dir",
        str(out_dir_path),
    ]

    env = os.environ.copy()
    src_dir = moonshot_root_path / "src"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_dir) if not existing else f"{src_dir}{os.pathsep}{existing}"

    subprocess.run(cmd, check=True, cwd=str(moonshot_root_path), env=env)


if __name__ == "__main__":
    # Adjust moonshot_root here if your clone lives elsewhere.
    prepare_moonshot_dataset(moonshot_root="C:/Users/andre/SMART-Moonshot")

