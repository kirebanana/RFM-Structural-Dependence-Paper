"""Streaming file and source identities for reproducible preliminary runs."""

import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path


def file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def source_identity(root):
    root = Path(root)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).splitlines()
    # Include unstaged/untracked source rather than pretending HEAD describes it.
    listed = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root, text=True
    ).splitlines()
    paths = [root / name for name in listed if name in ("pyproject.toml", "uv.lock") or (
        name.startswith(("src/", "experiments/", "scripts/", "vendor/"))
        and Path(name).suffix in (".py", ".rs", ".sh", ".toml", ".lock")
    )]
    identities = {str(path.relative_to(root)): file_identity(path)["sha256"] for path in sorted(paths)}
    digest = hashlib.sha256(repr(sorted(identities.items())).encode()).hexdigest()
    return {"revision": revision, "worktree_status": status, "source_sha256": digest}


def preprocessing_identity(pre_dir, dataset):
    directory = Path(pre_dir) / dataset
    required = ("meta.json", "table_info.json", "column_index.json", "relation_index.json", "nodes.rkyv", "offsets.rkyv", "p2f_adj.rkyv",
                "text_emb_all-MiniLM-L12-v2.bin")
    identities = {name: file_identity(directory / name) for name in required}
    if (directory / "text.json").is_file():
        identities["text.json"] = file_identity(directory / "text.json")
    return identities


def software_versions():
    versions = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "relbench", "relational-transformer"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions
