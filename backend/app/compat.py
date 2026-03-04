from __future__ import annotations

import sys
import types


def ensure_torchvision_functional_tensor() -> None:
    """
    Provide a compatibility shim for libraries that still import
    `torchvision.transforms.functional_tensor`.

    Newer torchvision releases expose the same symbols via
    `torchvision.transforms.functional`.
    """
    module_name = "torchvision.transforms.functional_tensor"
    if module_name in sys.modules:
        return

    try:
        from torchvision.transforms import functional as tv_functional
    except Exception:
        return

    shim = types.ModuleType(module_name)
    for attr in dir(tv_functional):
        try:
            setattr(shim, attr, getattr(tv_functional, attr))
        except Exception:
            continue
    sys.modules[module_name] = shim
