import os
import torch

LANGUAGES_NAMES = {
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ml": "Malayalam",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sl": "Slovenian",
    "sv": "Swedish",
    "tr": "Turkish",
    "zh": "Chinese simplified",
}

TTS_LANGUAGES = [
    "en",
    "de",
    "es",
    "fr",
    "hi",
    "it",
    "ja",
    "ko",
    "pl",
    "pt",
    "ru",
    "tr",
    "zh",
]

TTS_VC_LANGUAGES = [
    "en",
    "de",
    "ja",
    "pl",
    "pt",
    "tr",
]

NLTK_LANGUAGES = {
    "en": "english",
    "cs": "czech",
    "da": "danish",
    "nl": "dutch",
    "et": "estonian",
    "fi": "finish",
    "fr": "french",
    "de": "german",
    "el": "greek",
    "it": "italian",
    "ml": "malayalam",
    "no": "norwegian",
    "pl": "polish",
    "pt": "portuguese",
    "ru": "russian",
    "sl": "slovene",
    "es": "spanish",
    "sv": "swedish",
    "tr": "turkish",
}

DEFAULT_DEVICE = torch.device("cpu")

default_cache_folder = os.path.join(os.path.expanduser("~"), ".cache")
cache_folder = os.path.join(os.getenv("XDG_CACHE_HOME", default_cache_folder), "meow")

def set_cache_folder(path):
    global cache_folder
    cache_folder = path

def get_cache_folder():
    global cache_folder
    return cache_folder

def pbar_update(pbar_callback, value=0, total=0):
    if pbar_callback is not None:
        pbar_callback(value=value, total=total)

try:
    import xformers
    import xformers.ops
    XFORMERS_ATTENTION_AVAILABLE = True
    try:
        XFORMERS_ATTENTION_AVAILABLE = xformers._has_cpp_library
    except:
        pass
except:
    XFORMERS_ATTENTION_AVAILABLE = False

PYTORCH_ATTENTION_AVAILABLE = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

import nltk

def sentence_split(text, language="en", max_characters=0, extra_separators=[]):
    language = NLTK_LANGUAGES[language]

    text = text.lstrip().rstrip()

    texts = nltk.sent_tokenize(text, language=language)

    def normalize_texts(texts, separator):
        len_separator = len(separator)
        normalized_texts = []
        for text in texts:
            if max_characters == 0 or len(nltk.Text(text)) <= max_characters:
                normalized_texts.append(text)
            else:
                pos = 0
                for idx in range(len(text) - 1, -1, -1):
                    pos = text.find(separator, idx)
                    if pos != -1 and (max_characters == 0 or pos + len_separator <= max_characters):
                        pos = pos + len_separator
                        break

                if pos > 0:
                    normalized_texts.append(text[:pos].lstrip().rstrip())
                    normalized_texts.append(text[pos:].lstrip().rstrip())
                else:
                    normalized_texts.append(text)
        return normalized_texts

    def parse_newline(texts):
        chunks = []
        for text in texts:
            for line in text.split("\n"):
                line = line.lstrip().rstrip()
                if line != "":
                    chunks.append(line)
        return chunks

    def join_texts(texts, separator=" "):
        chunks = []
        token_counter = 0
        for text in texts:
            current_tokens = len(nltk.Text(text))
            extra_tokens = 0 if len(chunks) == 0 else len(separator)
            if max_characters == 0 or token_counter + current_tokens + extra_tokens <= max_characters:
                if len(chunks) == 0:
                    chunks = [text]
                else:
                    chunks[-1] = separator.join([chunks[-1], text])
                token_counter = token_counter + current_tokens + extra_tokens
            else:
                chunks.append(text)
                token_counter = current_tokens
        return chunks

    texts = parse_newline(texts)

    for separator in extra_separators:
        texts = normalize_texts(texts, separator)

    texts = normalize_texts(texts, " ")

    if max_characters != 0:
        texts = join_texts(texts, " ")

    return texts

