from omegaconf import DictConfig


def flatten_omegaconf(config: DictConfig) -> dict:
    """Flattens a nested OmegaConf object into a flat dictionary."""
    flat_dict = {}

    def flatten(d, parent_key="", sep="."):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, DictConfig):
                flatten(v, new_key, sep=sep)
            else:
                flat_dict[new_key] = v

    flatten(config)
    return flat_dict
