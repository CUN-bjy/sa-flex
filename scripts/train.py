from typing import Optional

import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision import transforms

from slot_attention.data import CLEVRDataModule, CLEVRwithMaskDataModule
from slot_attention.method import SlotAttentionMethod
from slot_attention.model import SlotAttentionModel
from slot_attention.params import SlotAttentionParams
from slot_attention.utils import ImageLogCallback
from slot_attention.utils import rescale

from datetime import datetime

def get_clevr_dataset(params):
    clevr_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(rescale),  # rescale between -1 and 1
            transforms.Resize(params.resolution),
        ]
    )

    return CLEVRDataModule(
        data_root=params.data_root,
        max_n_objects=10,
        train_batch_size=params.batch_size,
        val_batch_size=params.val_batch_size,
        clevr_transforms=clevr_transforms,
        num_train_images=params.num_train_images,
        num_val_images=params.num_val_images,
        num_workers=params.num_workers,
    )
    
def get_clevr_with_mask_dataset(params):
    clevr_transforms = {
        'image': transforms.Compose([
            transforms.CenterCrop(192),
            transforms.Resize(128, transforms.InterpolationMode.NEAREST)
        ]),     
        'mask': transforms.Compose([
            transforms.CenterCrop(192),
            transforms.Resize(128, transforms.InterpolationMode.NEAREST)
        ])
    }

    return CLEVRwithMaskDataModule(
        data_root=params.data_root,
        train_batch_size=params.batch_size,
        val_batch_size=params.val_batch_size,
        clevr_transforms=clevr_transforms,
        num_train_images=params.num_train_images,
        num_val_images=params.num_val_images,
        num_workers=params.num_workers,
    )

def main(params: Optional[SlotAttentionParams] = None):
    if params is None:
        params = SlotAttentionParams()

    assert params.num_slots > 1, "Must have at least 2 slots."

    if params.is_verbose:
        if params.num_train_images:
            print(f"INFO: restricting the train dataset size to `num_train_images`: {params.num_train_images}")
        if params.num_val_images:
            print(f"INFO: restricting the validation dataset size to `num_val_images`: {params.num_val_images}")

    clevr_datamodule = get_clevr_dataset(params) if not params.clevr_with_mask else get_clevr_with_mask_dataset()

    model = SlotAttentionModel(
        resolution=params.resolution,
        num_slots=params.num_slots,
        num_iterations=params.num_iterations,
        empty_cache=params.empty_cache,
    )

    method = SlotAttentionMethod(model=model, datamodule=clevr_datamodule, params=params)

    logger_name = f"{datetime.today()}-sa-clevr-n5"
    logger = pl_loggers.WandbLogger(project="sa-flex", name=logger_name)

    trainer = Trainer(
        logger=logger if params.is_logger_enabled else False,
        accelerator="cuda",
        strategy="ddp" if params.gpus > 1 else None,
        num_sanity_val_steps=params.num_sanity_val_steps,
        devices=params.gpus,
        max_epochs=params.max_epochs,
        log_every_n_steps=50,
        callbacks=[LearningRateMonitor("step"), ImageLogCallback(),] if params.is_logger_enabled else [],
    )
    trainer.fit(method, clevr_datamodule.train_dataloader(), clevr_datamodule.val_dataloader())


if __name__ == "__main__":
    main()