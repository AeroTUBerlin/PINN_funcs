import json
import os

from Rhombus_Euler_rxy_cflearn import main

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CONFIG_DIR = os.path.join(REPO_ROOT, 'examples', 'configs')

CONFIG_NAMES = [
    "rhombus_euler_paper_baseline",
    "rhombus_euler_paper_PIQS_off",
    "rhombus_euler_paper_PIQS_on",
]


def _resolve(path):
    """Resolve a config-relative path against the repo root, independent of the caller's cwd."""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


if __name__ == "__main__":
    for config_name in CONFIG_NAMES:
        with open(os.path.join(CONFIG_DIR, f"{config_name}.json")) as f:
            config_dict = json.load(f)

        bundle_path = _resolve(config_dict["data"]["bundle_path"])
        if not os.path.isfile(bundle_path):
            raise FileNotFoundError(
                f"Data bundle not found at {bundle_path}. Run `python helpers/fetch_data.py` first."
            )
        config_dict["data"]["bundle_path"] = bundle_path
        config_dict["data"]["save_path"] = _resolve(config_dict["data"]["save_path"])

        print(f"Running {config_name}...")
        main(
            config_dict["data"],
            config_dict["nn_training"],
            config_dict["nn_args"],
            comment=config_dict["comment"],
            protocol=_resolve(config_dict["protocol"]),
        )
