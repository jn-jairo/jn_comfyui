import os
import importlib
import traceback
import logging

MODULE_PATH = os.path.dirname(os.path.abspath(__file__))

try:
    from spandrel_extra_arches import EXTRA_REGISTRY
    from spandrel import MAIN_REGISTRY
    MAIN_REGISTRY.add(*EXTRA_REGISTRY)
    logging.info("Successfully imported spandrel_extra_arches: support for non commercial models.")
except:
    pass

import folder_paths

def make_dir(name, base_dir):
    if name not in folder_paths.folder_names_and_paths:
        new_dir = os.path.join(base_dir, name)
        os.makedirs(new_dir, exist_ok=True)
        folder_paths.add_model_folder_path(name, new_dir)

def make_dirs(names, base_dir):
    for name in names:
        make_dir(name, base_dir)

MODELS_DIRS = [
    "meow",
    "facerestore_models",
    "facedetection",
]

make_dirs(MODELS_DIRS, folder_paths.models_dir)

# NLTK
os.environ["NLTK_DATA"] = os.path.join(folder_paths.get_folder_paths("meow")[0], "nltk")
os.makedirs(os.environ["NLTK_DATA"], exist_ok=True)
import nltk
try:
    nltk.sent_tokenize("english", language="english")
except:
    nltk.download("punkt")

def load_nodes():
    error_messages = []
    node_class_mappings = {}
    node_display_name_mappings = {}

    for filename in sorted(os.listdir(os.path.join(MODULE_PATH, "nodes"))):
        module_name, extension = os.path.splitext(filename)

        if extension not in ["", ".py"] or module_name.startswith("__"):
            continue

        try:
            module = importlib.import_module(
                f".nodes.{module_name}", package=__package__
            )

            if hasattr(module, "NODE_CLASS_MAPPINGS"):
                node_class_mappings.update(getattr(module, "NODE_CLASS_MAPPINGS"))

            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                node_display_name_mappings.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS"))

            logging.debug(f"Imported '{module_name}' nodes")

        except Exception:
            error_message = traceback.format_exc()
            error_messages.append(f"Failed to import module '{module_name}' because {error_message}")

    if len(error_messages) > 0:
        logging.warning(
            f"Some nodes failed to load:\n\n"
            + "\n".join(error_messages)
        )

    return node_class_mappings, node_display_name_mappings

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = load_nodes()

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
