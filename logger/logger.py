from argparse import Namespace
from typing import (
    Optional,
    List,
    Dict,
    Any,
    Union,
    Iterable,
)

from pytorch_lightning.logging.neptune import NeptuneLogger

from pytorch_lightning.utilities import rank_zero_only


class CustomNeptuneLogger(NeptuneLogger):
    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        pass
