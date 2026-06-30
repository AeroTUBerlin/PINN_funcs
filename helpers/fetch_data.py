"""Download data bundles required to reproduce the examples from GitHub Releases.

Usage:
    python helpers/fetch_data.py [bundle_name ...]

With no arguments, downloads all registered bundles that aren't already present
and valid. Existing files are kept if their checksum already matches.
"""
import hashlib
import os
import sys
import urllib.request

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RELEASE_BASE_URL = "https://github.com/AeroTUBerlin/PINN_funcs/releases/download"

BUNDLES = {
    "rhombus_euler_bundle_v1": {
        "release_tag": "v0.1.0",
        "asset_name": "rhombus_euler_bundle_v1.npz",
        "dest": os.path.join(REPO_ROOT, "examples", "data", "rhombus", "rhombus_euler_bundle_v1.npz"),
        "sha256": "70e042ed08f0dfcf8227f86d15f3e7db915c3f7b5d780744a08c731130445541",
    },
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name, spec):
    dest = spec["dest"]
    if os.path.isfile(dest) and _sha256(dest) == spec["sha256"]:
        print(f"[{name}] already present and verified at {dest}")
        return

    url = f"{RELEASE_BASE_URL}/{spec['release_tag']}/{spec['asset_name']}"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp_path = dest + ".part"

    print(f"[{name}] downloading {url} ...")
    urllib.request.urlretrieve(url, tmp_path)

    actual_sha256 = _sha256(tmp_path)
    if actual_sha256 != spec["sha256"]:
        os.remove(tmp_path)
        raise ValueError(
            f"[{name}] checksum mismatch: expected {spec['sha256']}, got {actual_sha256}. Download corrupted or stale."
        )

    os.replace(tmp_path, dest)
    print(f"[{name}] saved to {dest}")


if __name__ == "__main__":
    requested = sys.argv[1:] or list(BUNDLES.keys())
    for bundle_name in requested:
        if bundle_name not in BUNDLES:
            raise SystemExit(f"Unknown bundle '{bundle_name}'. Known bundles: {list(BUNDLES.keys())}")
        fetch(bundle_name, BUNDLES[bundle_name])
