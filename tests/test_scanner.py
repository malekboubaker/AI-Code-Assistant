from pathlib import Path

from backend.rag.scanner import scan_project_with_stats


def test_scanner_includes_project_files_and_skips_heavy_folders(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.rs").write_text("pub fn run() {}\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    (tmp_path / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.ts").write_text("export const ignored = true;\n", encoding="utf-8")
    (tmp_path / "large.json").write_text("x" * 1_000_001, encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\x00")
    (tmp_path / ".env").write_text("SECRET=value\n", encoding="utf-8")

    stats = scan_project_with_stats(str(tmp_path))
    relative_paths = {path.relative_to(tmp_path).as_posix() for path in stats.files}

    assert "src/lib.rs" in relative_paths
    assert "README.md" in relative_paths
    assert "docker-compose.yml" in relative_paths
    assert "node_modules/ignored.ts" not in relative_paths
    assert "large.json" not in relative_paths
    assert "image.png" not in relative_paths
    assert ".env" not in relative_paths
    assert stats.languages["rust"] == 1
    assert stats.skipped_by_reason["too_large"] == 1
    assert stats.skipped_by_reason["secret"] == 1
