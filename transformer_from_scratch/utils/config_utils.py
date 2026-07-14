
import argparse
import os
from collections.abc import MutableMapping
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf, open_dict

from transformer_from_scratch.logger import logger

class ConfigNamespace(SimpleNamespace, MutableMapping):
    """SimpleNamespace that also behaves like a dict."""

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        delattr(self, key)

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()
    
    def __repr__(self):
        return f"ConfigNamespace({self.__dict__})"

    @classmethod
    def from_dict(cls, obj):
        """Recursively build ConfigNamespace from a nested dict."""
        if not isinstance(obj, dict):
            return obj
        return cls(**{k: cls.from_dict(v) for k, v in obj.items()})

    @classmethod
    def to_builtin(cls, obj):
        """
        Recursively convert a ConfigNamespace or nested structures into
        plain Python types (dicts, lists, primitives).

        Args:
            obj: ConfigNamespace, dict, list, tuple, or primitive.

        Returns:
            A fully Python-native structure, safe for serialization or logging.
        """
        if isinstance(obj, cls):
            return {k: cls.to_builtin(v) for k, v in vars(obj).items()}
        elif isinstance(obj, dict):
            return {k: cls.to_builtin(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [cls.to_builtin(v) for v in obj]
        else:
            return obj

def compose_config(
    overrides: Optional[List[str]] = None,
    config_dir: Optional[Path] = None,
) -> DictConfig:
    """Compose the project configuration using OmegaConf only.

    The ``defaults`` list in ``config.yaml`` selects configuration groups such
    as the model, optimizer, and loss. The ``_self_`` entry determines where
    the contents of ``config.yaml`` are inserted in the merge order.

    Group selections and value overrides can both be supplied from the CLI::

        optimizer=sgd training.epochs=20

    Args:
        overrides: OmegaConf overrides received from the command line.
        config_dir: Directory containing ``config.yaml`` and its config groups.
            Defaults to the repository's ``conf`` directory.

    Returns:
        The composed configuration as an OmegaConf ``DictConfig``.
    """
    config_dir = config_dir or Path(__file__).resolve().parents[2] / "conf"
    root_config = OmegaConf.load(config_dir / "config.yaml")

    defaults = list(root_config.get("defaults", []))
    with open_dict(root_config):
        del root_config["defaults"]

    default_groups = {
        group
        for item in defaults
        if isinstance(item, DictConfig)
        for group in item.keys()
    }

    group_overrides = {}
    value_overrides = []

    for override in overrides or []:
        key, separator, value = override.partition("=")

        if separator and key in default_groups:
            group_overrides[key] = value
        else:
            value_overrides.append(override)

    config_parts = []
    root_config_added = False

    for default in defaults:
        if default == "_self_":
            config_parts.append(root_config)
            root_config_added = True
            continue

        if not isinstance(default, DictConfig):
            raise ValueError(f"Unsupported defaults entry: {default!r}")

        group, default_name = next(iter(default.items()))
        selected_name = group_overrides.get(group, default_name)
        group_path = config_dir / group / f"{selected_name}.yaml"

        if not group_path.is_file():
            raise FileNotFoundError(
                f"Configuration '{selected_name}' was not found in group '{group}'."
            )

        group_config = OmegaConf.load(group_path)
        if group in group_config:
            config_parts.append(group_config)
        else:
            config_parts.append(OmegaConf.create({group: group_config}))

    if not root_config_added:
        config_parts.append(root_config)

    if value_overrides:
        config_parts.append(OmegaConf.from_dotlist(value_overrides))

    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver(
            "now",
            lambda date_format: datetime.now().strftime(date_format),
        )

    return OmegaConf.merge(*config_parts)


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """
    Recursively flattens a nested dictionary into dot notation.
    Example:
        {"a": {"b": 1}} → {"a.b": 1}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            if isinstance(v, (list, tuple)):
                v = ",".join(map(str, v))
            elif not isinstance(v, (str, int, float, bool)) and v is not None:
                v = str(v)
            items.append((new_key, v))
    return dict(items)


def load_environment(args: Optional[argparse.Namespace] = None) -> None:
    """
    Loads environment variables from the project's .env file. Also exports
    external CLI arguments to environment variables when provided.

    - Ensures predictable .env resolution.
    - Warns if the .env file is missing.
    - Avoids silent failures by logging explicit outcomes.

    Args:
        args (Optional[argparse.Namespace]): Parsed CLI arguments to export.
    """
    env_path = Path(__file__).resolve().parents[2] / ".env"

    # --- Load .env file ---
    if env_path.is_file():
        try:
            load_dotenv(dotenv_path=env_path, override=True)
            logger.info(f"Loaded environment variables from: {env_path}")
        except Exception as e:
            logger.warning(f"Failed to load .env file at {env_path}: {e}")
    else:
        logger.warning(f".env file not found at expected path: {env_path}")

    # --- Export external CLI args to environment ---
    if args is not None:
        try:
            export_args_to_env(args)
            logger.info("Exported CLI arguments into environment variables.")
        except Exception as e:
            logger.warning(f"Failed to export CLI args to environment: {e}")


def  parse_args() -> argparse.Namespace:
    """
    Parse external CLI arguments that remain outside the YAML configuration.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="CLI flags for external runtime behavior."
    )

    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Enable a minimal debugging execution path."
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="RUN_ID",
        help="MLflow run ID to resume training from."
    )

    args, unknown = parser.parse_known_args()
    return args, unknown


def export_args_to_env(args: argparse.Namespace) -> None:
    """
    Export parsed CLI arguments into environment variables so the training
    pipeline can inspect them independently from the YAML configuration.

    Args:
        args (argparse.Namespace): Arguments parsed by parse_args().
    """
    os.environ["FAST_DEV_RUN"] = "1" if args.fast_dev_run else "0"

    if args.resume is not None:
        os.environ["RESUME_RUN_ID"] = args.resume
    else:
        # Ensure the variable is absent if no resume ID was provided.
        os.environ.pop("RESUME_RUN_ID", None)
