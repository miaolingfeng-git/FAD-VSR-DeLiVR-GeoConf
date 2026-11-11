from .network import FADVSR, Network


def make(spec, load_sd: bool = False):
    spec = dict(spec)
    name = spec.pop("name", "network")
    args = spec.pop("args", {})
    if name not in {"network", "fadv_sr", "fadv_sr_geoconf", "fadvSR", "FADVSR"}:
        raise ValueError(f"unknown model: {name}")
    model = FADVSR(**args)
    if load_sd and "sd" in spec:
        model.load_state_dict(spec["sd"], strict=False)
    return model
