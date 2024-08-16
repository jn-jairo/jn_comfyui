import os
import shutil
import urllib.request
import huggingface_hub
import logging

from abc import abstractmethod

from .utils import get_cache_folder, DEFAULT_DEVICE

def get_model_loader(info, device=None, download=True, base_dir=None):
    loader = info["loader"]

    if not loader:
        raise NotImplementedError(f"Model '{name}' not implemented")

    return loader(info=info, device=device, download=download, base_dir=base_dir)

class BaseModel:

    def __init__(self, info, device=None, download=True, base_dir=None, *args, **kwargs):
        self.name = info["name"]
        self.info = info
        self.device = DEFAULT_DEVICE if device is None else device
        self.download = download
        self.base_dir = get_cache_folder() if base_dir is None else base_dir

        os.makedirs(self.base_dir, exist_ok=True)

        if "filename" in self.info:
            self.file_path = os.path.join(base_dir, info["filename"])
        else:
            self.file_path = None

        self.dependencies = {}

        if "dependencies" in self.info:
            for k, v in self.info["dependencies"].items():
                self.dependencies[k] = get_model_loader(v, device=self.device, download=self.download, base_dir=self.base_dir)

    def download_file(self):
        # raise NotImplementedError("Download model file not implemented")

        logging.info(f"Downloading model '{self.name}'")

        if self.info["type"] == "hf":
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            huggingface_hub.hf_hub_download(self.info["repo_id"], self.info["repo_path"], local_dir=self.base_dir)
            shutil.move(os.path.join(self.base_dir, self.info["repo_path"]), self.file_path)
        elif self.info["type"] == "url":
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            urllib.request.urlretrieve(self.info["url"], self.file_path)
        else:
            raise RuntimeError("Invalid model download method: " + self.info["type"])

        logging.info(f"Downloading model '{self.name}' OK")

    def get_file_path(self):
        if self.file_path is not None and self.download and not os.path.exists(self.file_path):
            self.download_file()

        return self.file_path

    def load(self, *args, **kwargs):
        for k in self.dependencies.keys():
            self.dependencies[k].load()

        logging.info(f"Loading model '{self.name}'")
        self.load_model(*args, **kwargs)
        self.to(self.device)
        logging.info(f"Loading model '{self.name}' OK")

    @abstractmethod
    def load_model(self, *args, **kwargs):
        pass

    def to(self, device):
        if hasattr(self, "model") and self.model is not None and hasattr(self.model, "to"):
            log = next(self.model.parameters()).device != device if hasattr(self.model, "parameters") else True
            if log:
                logging.info(f"Model '{self.name}' to " + repr(device))
            self.model.to(device)

        if hasattr(self, "model_to"):
            self.model_to(device)

        for k in self.dependencies.keys():
            self.dependencies[k].to(device)

    def execute(self, *args, **kwargs):
        pass

    def encode(self, *args, **kwargs):
        pass

    def decode(self, *args, **kwargs):
        pass

class ModelDeviceContext:

    def __init__(self, model_loader, device=None):
        self.model_loader = model_loader
        self.device = device

    def __enter__(self):
        if self.device is not None:
            self.model_loader.to(self.device)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.device is not None:
            self.model_loader.to(self.model_loader.device)

