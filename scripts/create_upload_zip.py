from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ZIP_NAME = ROOT / "ai-code-assistant-phase2.zip"
PROJECT_DIR_NAME = "ai-code-assistant-phase2"


REQUIREMENTS_TEXT = """torch>=2.0.0
transformers>=4.36.0
datasets>=2.14.0
peft>=0.7.0
bitsandbytes>=0.41.0
numpy>=1.24.0
"""

CONFIG_TEXT = """model: Qwen/Qwen3-Coder-Next
max_seq_length: 8192
lora_rank: 32
lora_alpha: 64
epochs: 3
batch_size: 8
learning_rate: 0.0002
"""

README_TEXT = """# AI Code Assistant Phase 2

This package contains merged train, validation, and test datasets prepared for cloud fine-tuning.
The training and evaluation entrypoints are placeholders and will be completed in the next step.
"""


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def create_placeholder_script(path: Path, title: str) -> None:
    write_text(
        path,
        f'"""{title} placeholder."""\n\n# TODO: implement {title.lower()} logic here.\n',
    )


def copy_data_files(project_root: Path) -> None:
    target_data_dir = project_root / "data"
    target_data_dir.mkdir(parents=True, exist_ok=True)
    for name in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        source = DATA_DIR / name
        destination = target_data_dir / name
        if not source.exists():
            raise FileNotFoundError(f"Missing required merged file: {source}")
        shutil.copy2(source, destination)


def build_package_tree(tmp_root: Path) -> Path:
    project_root = tmp_root / PROJECT_DIR_NAME
    (project_root / "data").mkdir(parents=True, exist_ok=True)
    (project_root / "scripts").mkdir(parents=True, exist_ok=True)

    copy_data_files(project_root)
    create_placeholder_script(project_root / "scripts" / "train.py", "Training")
    create_placeholder_script(project_root / "scripts" / "evaluate.py", "Evaluation")
    write_text(project_root / "requirements.txt", REQUIREMENTS_TEXT)
    write_text(project_root / "config.yaml", CONFIG_TEXT)
    write_text(project_root / "README.md", README_TEXT)

    return project_root


def create_zip(project_root: Path, zip_path: Path) -> list[str]:
    files_inside: list[str] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for file_path in sorted(project_root.rglob("*")):
            if not file_path.is_file():
                continue
            arcname = file_path.relative_to(project_root.parent).as_posix()
            archive.write(file_path, arcname)
            files_inside.append(arcname)
    return files_inside


def main() -> int:
    if not (DATA_DIR / "train.jsonl").exists() or not (DATA_DIR / "val.jsonl").exists() or not (DATA_DIR / "test.jsonl").exists():
        print("❌ Missing merged data files. Run scripts/merge_multitask.py first.")
        return 1

    with tempfile.TemporaryDirectory(prefix="phase2_zip_") as tmp_dir:
        project_root = build_package_tree(Path(tmp_dir))
        files_inside = create_zip(project_root, ZIP_NAME)

    zip_size = ZIP_NAME.stat().st_size if ZIP_NAME.exists() else 0
    zip_size_mb = zip_size / (1024 * 1024)

    print("=== ZIP PACKAGE CREATED ===")
    print(f"ZIP file path: {ZIP_NAME}")
    print(f"ZIP file size: {zip_size_mb:.2f} MB")
    print("Files inside:")
    for item in files_inside:
        print(f"  - {item}")
    print("Ready to upload to Azure Blob Storage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())