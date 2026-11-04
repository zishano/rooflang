"""Run the existing layer analyzer on a model prefix ending at target layer.

This avoids simulating later decoder layers when only one layer's timing is
needed. The target layer and all earlier layers retain their original model
configuration and graph construction.
"""
import importlib
import runpy
import sys
from pathlib import Path


def main():
    args = sys.argv[1:]
    try:
        layer = int(args[args.index("--layer") + 1])
    except (ValueError, IndexError):
        raise SystemExit("--layer N is required")
    if layer < 0:
        raise SystemExit("--layer must be non-negative")

    prefix_len = layer + 1
    package = importlib.import_module("rooflang.programs.models.dsv4_flash")
    config = importlib.import_module("rooflang.programs.models.dsv4_flash.config")
    model_impl = importlib.import_module("rooflang.programs.models.dsv4_flash.model")
    optim_impl = importlib.import_module(
        "rooflang.programs.models.dsv4_flash.optimization")
    if prefix_len > len(config.COMPRESS_RATIOS):
        raise SystemExit(f"layer {layer} exceeds configured model depth")

    ratios = tuple(config.COMPRESS_RATIOS[:prefix_len])
    for module in (package, model_impl, optim_impl):
        module.N_LAYERS = prefix_len
        module.COMPRESS_RATIOS = ratios

    analyzer = Path(__file__).with_name("single_layer_timing.py")
    sys.argv = [str(analyzer), *args]
    runpy.run_path(str(analyzer), run_name="__main__")


if __name__ == "__main__":
    main()
