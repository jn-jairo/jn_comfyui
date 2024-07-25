import os
import yaml
import folder_paths

def merge_dicts(d1, d2):
    merged = dict(d1)  # Start with a copy of the first dictionary
    for k, v in d2.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = merge_dicts(merged[k], v)  # Recursively merge nested dictionaries
        elif v is not None:
            merged[k] = v  # Otherwise, just set the value
    return merged

def load_config_file(yaml_path):
    config = {}
    if os.path.isfile(yaml_path):
        with open(yaml_path, 'r') as stream:
            config = yaml.safe_load(stream)
    return config

def load_config():
    default_config = {
        "preview_device": None,
        "extension_device": {},
        "memory_estimation_multiplier": -1,
        "temperature": {
            "limit": {
                "execution": {
                    "safe": 0,
                    "max": 0,
                },
                "progress": {
                    "safe": 0,
                    "max": 0,
                }
            },
            "cool_down": {
                "each": 5,
                "max": 0
            }
        },
    }

    user_config = load_config_file(os.path.join(folder_paths.base_path, "jncomfy.yaml"))

    config = merge_dicts(default_config, user_config)

    return config

config = load_config()
