"""Pre-fetch the weights the request path loads, into the local model cache.

The request path reads its models offline: the deployment container has no
route to the hub, and a download inside a request would spend the whole
per-request budget before a Question was answered. So every model Settings
names is fetched here, once, and the cache is baked into the image.

    python -m scripts.fetch_models
    python -m scripts.fetch_models --all   # including what only measurement loads

Run it again after changing a model name in Settings — a stale cache is what
the offline load fails on, loudly, at startup.
"""

import argparse

from faster_whisper.utils import download_model
from huggingface_hub import snapshot_download

from medapp.config import Settings
from medapp.config import settings as default_settings

# What a cross-encoder needs to load: the config, the tokenizer and one copy of
# the weights. The `.bin` duplicate of the safetensors is excluded — it is the
# same parameters again, and doubles both the download and the image.
WEIGHT_PATTERNS = ["*.json", "*.txt", "*.model", "*.safetensors"]


def models_to_fetch(settings: Settings) -> tuple[str, ...]:
    """The hub repositories to fill the cache with.

    Every model the request path loads and no others: the reranker, which ranks
    a Conversation's sentences, and the Entailment judge behind it.

    Args:
        settings: The resolved environment.
    """
    return (settings.rerank_model, settings.nli_model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    print(f"{default_settings.whisper_model} -> {default_settings.model_cache_dir}")
    path = download_model(
        default_settings.whisper_model,
        cache_dir=str(default_settings.model_cache_dir),
    )
    print(f"  {path}")

    for repository in models_to_fetch(default_settings):
        print(f"{repository} -> {default_settings.model_cache_dir}")
        path = snapshot_download(
            repository,
            cache_dir=str(default_settings.model_cache_dir),
            allow_patterns=WEIGHT_PATTERNS,
        )
        print(f"  {path}")


if __name__ == "__main__":
    main()
