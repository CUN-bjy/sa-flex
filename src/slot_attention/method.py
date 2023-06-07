import pytorch_lightning as pl
import torch
from torch import optim
from torchvision import utils as vutils
from torch.nn import functional as F
import numpy as np

from slot_attention.model import SlotAttentionModel
from slot_attention.params import SlotAttentionParams
from slot_attention.utils import Tensor
from slot_attention.utils import to_rgb_from_tensor

from slot_attention.evaluator import adjusted_rand_index


class SlotAttentionMethod(pl.LightningModule):
    def __init__(self, model: SlotAttentionModel, datamodule: pl.LightningDataModule, params: SlotAttentionParams):
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.params = params
        self.validation_step_outputs = []
        self.automatic_optimization = False

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        opt_sa = self.optimizers()
        sh_sa = self.lr_schedulers()
        opt_sa.zero_grad()
        
        if self.params.clevr_with_mask:
            batch, gt_masks = batch["image"], batch["mask"]
        
        train_loss = self.model.loss_function(batch)
        
        self.manual_backward(train_loss["loss"])
        opt_sa.step()
        sh_sa.step()
        
        logs = {key: val.item() for key, val in train_loss.items()}
        self.log_dict(logs, sync_dist=True)

    def sample_images(self):
        dl = self.datamodule.val_dataloader()
        perm = torch.randperm(self.params.batch_size)
        idx = perm[: self.params.n_samples]
        if self.params.clevr_with_mask:
            batch = next(iter(dl))['image'][idx]
        else:
            batch = next(iter(dl))[idx]
            
        if self.params.gpus > 0:
            batch = batch.to(self.device)

        recon_combined, recons, masks, slots = self.model.forward(batch)

        # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    batch.unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recons * masks + (1 - masks),  # each slot
                ],
                dim=1,
            )
        )

        batch_size, num_slots, C, H, W = recons.shape
        images = vutils.make_grid(
            out.view(batch_size * out.shape[1], C, H, W).cpu(), normalize=False, nrow=out.shape[1],
        )

        return images

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        if self.params.clevr_with_mask:
            batch, gt_masks = batch["image"], batch["mask"]
        
        recon_combined, recons, masks, slots = self.model.forward(batch)
        
        mse_loss = F.mse_loss(recon_combined, batch)
        sparse_loss = torch.mean(torch.abs(F.relu(slots)))
        val_loss = mse_loss + self.params.reg_weight * sparse_loss
        
        if self.params.clevr_with_mask:    
            ari = adjusted_rand_index(gt_masks, masks, exclude_background=False).mean()
            fgari = adjusted_rand_index(gt_masks, masks).mean()
            self.validation_step_outputs.append({"loss": val_loss.item(), "sparse_loss": sparse_loss.item(), "mse_loss": mse_loss.item(), "ARI": ari.item(), "FG-ARI": fgari.item()})
        else:
            self.validation_step_outputs.append({"loss": val_loss.item(), "sparse_loss": sparse_loss.item(), "mse_loss": mse_loss.item(),})
        return val_loss

    def on_validation_epoch_end(self):
        avg_loss = np.array([x["loss"] for x in self.validation_step_outputs]).mean()
        avg_sparse_loss = np.array([x["sparse_loss"] for x in self.validation_step_outputs]).mean()
        avg_mse_loss = np.array([x["mse_loss"] for x in self.validation_step_outputs]).mean()
        avg_ari = np.array([x["ARI"] for x in self.validation_step_outputs]).mean()
        avg_fgari = np.array([x["FG-ARI"] for x in self.validation_step_outputs]).mean()
        logs = {
            "avg_val_loss": avg_loss,
            "avg_sparse_loss": avg_sparse_loss,
            "avg_mse_loss": avg_mse_loss,
            "avg_ari": avg_ari,
            "avr_fgari": avg_fgari,
        }
        self.log_dict(logs, sync_dist=True)
        print("; ".join([f"{k}: {v:.6f}" for k, v in logs.items()]))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)

        warmup_steps_pct = self.params.warmup_steps_pct
        decay_steps_pct = self.params.decay_steps_pct
        total_steps = self.params.max_epochs * len(self.datamodule.train_dataloader())

        def warm_and_decay_lr_scheduler(step: int):
            warmup_steps = warmup_steps_pct * total_steps
            decay_steps = decay_steps_pct * total_steps
            assert step < total_steps
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            factor *= self.params.scheduler_gamma ** (step / decay_steps)
            return factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler)

        return (
            [optimizer],
            [scheduler],
        )