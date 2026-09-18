"""Pre-fetch the weights the request path loads, into the local model cache.

The request path reads its models offline: the deployment container has no
route to the hub, and a download inside a request would spend the whole
per-request budget before a Question was answered. So every model Settings
names is fetched here, once, and the cache is baked into the image.

    python -m scripts.fetch_models

Run it again after changing a model name in Settings — a stale cache is what
the offline load fails on, loudly, at startup.
"""

from huggingface_hub import snapshot_download

from medapp.config import Settings, settings

# What a cross-encoder needs to load: the config, the tokenizer and one copy of
# the weights. The `.bin` duplicate of the safetensors is excluded — it is the
# same parameters again, and doubles both the download and the image.
WEIGHT_PATTERNS = ["*.json", "*.txt", "*.model", "*.safetensors"]


def models_to_fetch(settings: Settings) -> tuple[str, ...]:
    """The hub repositories the current configuration loads.

    Only the reranker so far: the Entailment and dense models are named in
    Settings but no component loads them yet, and fetching weights nothing
    reads would put gigabytes into the image for nothing.
    """
    return (settings.rerank_model,)


def main() -> None:
    for repository in models_to_fetch(settings):
        print(f"{repository} -> {settings.model_cache_dir}")
        path = snapshot_download(
            repository,
            cache_dir=str(settings.model_cache_dir),
            allow_patterns=WEIGHT_PATTERNS,
        )
        print(f"  {path}")


if __name__ == "__main__":
    main()
