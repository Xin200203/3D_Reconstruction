from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner.checkpoint import CheckpointLoader
import os

@HOOKS.register_module()
class PartialLoadHook(Hook):
    """Load selected parameters from a checkpoint into specified submodule.

    Args:
        pretrained (str): Path to checkpoint.
        submodule (str): Attribute name of model to load (e.g. 'backbone3d').
        prefix_replace (Tuple[str,str], optional): regex replacement to align names.
        strict (bool): strict flag for load_state_dict.
    """

    def __init__(self, pretrained: str, submodule: str = 'backbone3d', prefix_replace=None, strict: bool = False):
        self.pretrained = pretrained
        self.submodule = submodule  # dot-separated path
        self.prefix_replace = prefix_replace
        self.strict = strict

    @staticmethod
    def _locate_module(model, path: str):
        """Traverse attributes by dot path to get nested submodule."""
        cur = model
        for name in path.split('.'):  # supports recursive modules
            if not hasattr(cur, name):
                return None
            cur = getattr(cur, name)
        return cur

    def before_train(self, runner):
        logger = runner.logger
        if not os.path.isfile(self.pretrained):
            logger.warning(f'PartialLoadHook: checkpoint {self.pretrained} not found, skip.')
            return

        model = runner.model
        module = self._locate_module(model, self.submodule)
        if module is None:
            logger.warning(f'PartialLoadHook: submodule {self.submodule} not found, skip.')
            return
        checkpoint = CheckpointLoader.load_checkpoint(self.pretrained, map_location='cpu', logger=logger)
        state_dict = checkpoint.get('state_dict', checkpoint)
        if self.prefix_replace is not None:
            import re
            new_state = {}
            pattern, repl = self.prefix_replace
            for k, v in state_dict.items():
                new_k = re.sub(pattern, repl, k)
                new_state[new_k] = v
            state_dict = new_state
        # filter keys belonging to module
        prefix = self.submodule + '.'
        filtered = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        missing, unexpected = module.load_state_dict(filtered, strict=self.strict)
        logger.info(f'PartialLoadHook loaded {len(filtered)} params into {self.submodule}, missing {len(missing)}, unexpected {len(unexpected)}') 