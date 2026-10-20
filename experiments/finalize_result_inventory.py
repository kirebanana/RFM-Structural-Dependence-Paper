"""Freeze final source identities and checksums without modifying the Git index."""
import argparse
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

from rfm_structure.provenance import file_identity, source_identity


def freeze(root, repository, amendment=None):
    if amendment is not None and (not amendment.replace("_", "").replace("-", "").isalnum()):
        raise ValueError("amendment must be a simple alphanumeric name")
    patch_name = f"source_{amendment}.patch" if amendment else "source_at_finalization.patch"
    identity_name = f"source_{amendment}_identity.json" if amendment else "source_identity.json"
    targets = (patch_name, identity_name, "checksums.json")
    if any((root / name).exists() for name in targets):
        raise FileExistsError("refusing to overwrite existing source snapshot or inventory")
    baseline = json.loads((root / "catalog.json").read_text())["baseline"]
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                                    cwd=repository, text=True).splitlines()
    selected = sorted(name for name in names if name in ("pyproject.toml", "uv.lock") or (
        name.startswith(("src/", "experiments/", "tests/", "scripts/", "vendor/"))
        and Path(name).suffix in (".py", ".rs", ".sh", ".toml", ".lock")))
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=repository, text=True).splitlines())
    delta = subprocess.check_output(["git", "diff", "--binary", baseline, "--", *[n for n in selected if n in tracked]],
                                    cwd=repository, text=True)
    for name in selected:
        if name not in tracked:
            content = (repository / name).read_text().splitlines(keepends=True)
            delta += f"diff --git a/{name} b/{name}\nnew file mode 100644\n"
            delta += "".join(difflib.unified_diff([], content, fromfile="/dev/null", tofile=f"b/{name}"))
    (root / patch_name).write_text(delta)
    identities = {name: {key: value for key, value in file_identity(repository / name).items() if key != "path"}
                  for name in selected}
    identity = {**source_identity(repository), "baseline": baseline, "files": identities,
                "scope": "current finalization source, not an exact reconstruction of each historical run",
                "amendment": amendment}
    (root / identity_name).write_text(json.dumps(identity, indent=2) + "\n")
    inventory = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            content = path.read_bytes()
            inventory[str(path.relative_to(root))] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    (root / "checksums.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(f"Inventoried {len(inventory)} files; Git index unchanged.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--amendment", help="new source snapshot name; preserve the original finalization snapshot")
    args = parser.parse_args()
    freeze(args.root, Path(__file__).resolve().parents[1], args.amendment)


if __name__ == "__main__":
    main()
