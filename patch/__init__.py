import os
import importlib
import traceback
import logging

MODULE_PATH = os.path.dirname(os.path.abspath(__file__))

LOG_PREFIX = "JNComfy: "

def load_patches():
    error_messages = []
    patches = {}

    for filename in sorted(os.listdir(MODULE_PATH)):
        module_name, extension = os.path.splitext(filename)

        if extension not in ["", ".py"] or module_name.startswith("__"):
            continue

        try:
            module = importlib.import_module(
                f"patch.{module_name}", package=__package__
            )

            if hasattr(module, "PATCHES"):
                patches.update(getattr(module, "PATCHES"))

            logging.debug(f"{LOG_PREFIX}Imported '{module_name}' patch")

        except Exception:
            error_message = traceback.format_exc()
            error_messages.append(f"Failed to import patch '{module_name}' because {error_message}")

    if len(error_messages) > 0:
        logging.warning(
            f"{LOG_PREFIX}Some patches failed to load:\n\n"
            + "\n".join(error_messages)
        )

    return patches

def apply_patches(config):
    patches = load_patches()

    for name in sorted(patches):
        logging.info(f"{LOG_PREFIX}Applying patch '{name}'")

        try:
            patches[name](config)

            logging.info(f"{LOG_PREFIX}Applying patch '{name}' APPLIED")
        except Exception:
            error_message = traceback.format_exc()
            logging.error(f"{LOG_PREFIX}Applying patch '{name}' FAILED:\n\n{error_message}")
