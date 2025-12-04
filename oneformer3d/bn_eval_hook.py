from __future__ import annotations

from typing import List
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class ModuleEvalFreezeHook(Hook):
    """Set selected submodules to eval() at training start.

    This prevents BatchNorm running stats from updating on frozen branches.

    Args:
        module_prefixes (List[str]): Module name prefixes to match
            (e.g., ["bi_encoder.backbone3d", "decoder"]).
        also_require_grad_off (bool): If True, additionally set
            requires_grad=False for matched modules' parameters.
        verbose (bool): Print matched module count to logger.
    """

    def __init__(self,
                 module_prefixes: List[str],
                 also_require_grad_off: bool = False,
                 verbose: bool = True) -> None:
        self.module_prefixes = list(module_prefixes)
        self.also_require_grad_off = also_require_grad_off
        self.verbose = verbose

    def before_train(self, runner) -> None:  # type: ignore[override]
        model = runner.model
        matched = 0
        for name, module in model.named_modules():
            if any(name.startswith(p) for p in self.module_prefixes):
                module.eval()
                if self.also_require_grad_off:
                    for p in module.parameters():
                        p.requires_grad = False
                matched += 1
        if self.verbose:
            runner.logger.info(
                f"[ModuleEvalFreezeHook] set eval() for {matched} modules matching prefixes: {self.module_prefixes}")

