from typing import Optional
from typing import Tuple

from dataset import DATASET_PATH
import attr


@attr.s(auto_attribs=True)
class SlotAttentionParams:
    lr: float = 0.0004
    batch_size: int = 10
    val_batch_size: int = 10
    resolution: Tuple[int, int] = (128, 128)
    num_slots: int = 5
    num_iterations: int = 3
    data_root: str = DATASET_PATH
    gpus: int = 2
    max_epochs: int = 50
    num_sanity_val_steps: int = 1
    scheduler_gamma: float = 0.5
    weight_decay: float = 0.0
    num_train_images: Optional[int] = None
    num_val_images: Optional[int] = None
    empty_cache: bool = True
    is_logger_enabled: bool = True
    is_verbose: bool = True
    num_workers: int = 8
    n_samples: int = 5
    warmup_steps_pct: float = 0.02
    decay_steps_pct: float = 0.2
    clevr_with_mask: bool = True
