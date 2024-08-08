import torch
import contextlib
import funcy

from ...base_model import BaseModel

if (
    torch.cuda.is_available() and
    hasattr(torch.cuda, "amp") and
    hasattr(torch.cuda.amp, "autocast") and
    hasattr(torch.cuda, "is_bf16_supported") and
    torch.cuda.is_bf16_supported()
):
    autocast = funcy.partial(torch.cuda.amp.autocast, dtype=torch.bfloat16)
else:
    @contextlib.contextmanager
    def autocast():
        yield

class InferenceContext:
    def __init__(self, benchmark=False):
        # we can't expect inputs to be the same length, so disable benchmarking by default
        self._chosen_cudnn_benchmark = benchmark
        self._cudnn_benchmark = None
        self._cuda_allow_tf32 = None
        self._cudnn_allow_tf32 = None

    def __enter__(self):
        self._cudnn_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = self._chosen_cudnn_benchmark

        if torch.cuda.is_available():
            self._cuda_allow_tf32 = torch.backends.cuda.matmul.allow_tf32 
            self._cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.backends.cudnn.benchmark = self._cudnn_benchmark

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = self._cuda_allow_tf32
            torch.backends.cudnn.allow_tf32 = self._cudnn_allow_tf32

@contextlib.contextmanager
def inference_mode():
    with InferenceContext(), torch.inference_mode(), torch.no_grad(), autocast():
        yield

class TtsBaseModel(BaseModel):
    CONTEXT_WINDOW_SIZE = 1024

    SEMANTIC_RATE_HZ = 49.9
    SEMANTIC_VOCAB_SIZE = 10_000

    CODEBOOK_SIZE = 1024
    N_COARSE_CODEBOOKS = 2
    N_FINE_CODEBOOKS = 8
    COARSE_RATE_HZ = 75

    SAMPLE_RATE = 24_000

    TEXT_ENCODING_OFFSET = 10_048
    SEMANTIC_PAD_TOKEN = 10_000
    TEXT_PAD_TOKEN = 129_595
    SEMANTIC_INFER_TOKEN = 129_599

    COARSE_SEMANTIC_PAD_TOKEN = 12_048
    COARSE_INFER_TOKEN = 12_050


    def __init__(self, model_class, config_class, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_class = model_class
        self.config_class = config_class
        self.model = None

    def load_model(self, *args, **kwargs):
        checkpoint = torch.load(self.get_file_path(), map_location=self.device)

        model_args = checkpoint["model_args"]

        # this is a hack
        if "input_vocab_size" not in model_args:
            model_args["input_vocab_size"] = model_args["vocab_size"]
            model_args["output_vocab_size"] = model_args["vocab_size"]
            del model_args["vocab_size"]

        conf = self.config_class(**checkpoint["model_args"])
        model = self.model_class(conf)
        state_dict = checkpoint["model"]

        # fixup checkpoint
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        extra_keys = set(state_dict.keys()) - set(model.state_dict().keys())
        extra_keys = set([k for k in extra_keys if not k.endswith(".attn.bias")])

        missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
        missing_keys = set([k for k in missing_keys if not k.endswith(".attn.bias")])

        if len(extra_keys) != 0:
            raise ValueError(f"extra keys found: {extra_keys}")

        if len(missing_keys) != 0:
            raise ValueError(f"missing keys: {missing_keys}")

        model.load_state_dict(state_dict, strict=False)

        n_params = model.get_num_params()
        val_loss = checkpoint["best_val_loss"].item()
        # logger.info(f"model loaded: {round(n_params/1e6,1)}M params, {round(val_loss,3)} loss")

        model.eval()
        # model.to(self.device)

        self.model = model

        del checkpoint, state_dict, model

