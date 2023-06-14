from typing import Optional
from typing import Tuple

from dataset import DATASET_PATH
import attr


@attr.s(auto_attribs=True)
class SlotAttentionParams:
    prefix: str = "sa-clevr-n5"
    seed: int = 0
    data_root: str = DATASET_PATH
    clevr_with_mask: bool = True
    gpus: int = 2
    num_workers: int = 8
    n_samples: int = 5
    
    # training parameters
    max_epochs: int = 70
    lr: float = 0.0004
    lr_d: float = 0.00005
    annealing_steps: int = 100000
    scheduler_gamma: float = 0.5
    weight_decay: float = 0.0
    warmup_steps_pct: float = 0.02
    decay_steps_pct: float = 0.2

    batch_size: int = 20
    val_batch_size: int = 20
    num_train_images: Optional[int] = None
    num_val_images: Optional[int] = None
    
    # slot parameters
    resolution: Tuple[int, int] = (128, 128)
    num_slots: int = 5
    slot_size: int = 64
    num_iterations: int = 3
    
    # sparse mask
    use_sparse_mask: bool = False
    hidden_mask_layer: int = 512
    feed_encoded_out: bool = False
    tau_annealing: bool = False
    
    # objective parameters
    wakeup_sparse_mask_pct: float = 0.0
    auto_sparse_weight: bool = False
    sparse_weight: float = 0.0
    tc_weight: float = 0.0
    
    num_sanity_val_steps: int = 1
    empty_cache: bool = True
    is_logger_enabled: bool = True
    is_verbose: bool = True
    
    # pretrain
    ckp_path: str = ""
    freeze_encoder: bool = False
