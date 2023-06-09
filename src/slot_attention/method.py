import pytorch_lightning as pl
import torch
from torch import optim
from torchvision import utils as vutils
from torch.nn import functional as F
import numpy as np

from slot_attention.model import SlotAttentionModel
from slot_attention.params import SlotAttentionParams
from slot_attention.utils import Tensor, to_rgb_from_tensor
from slot_attention.utils import permute_dims, linear_annealing, warm_and_decay_annealing

from slot_attention.evaluator import adjusted_rand_index
from slot_attention.discriminator import Discriminator


class SlotAttentionMethod(pl.LightningModule):
    def __init__(self, model: SlotAttentionModel, datamodule: pl.LightningDataModule, params: SlotAttentionParams):
        super().__init__()
        self.datamodule = datamodule
        self.params = params
        self.validation_step_outputs = []
        self.activate_mask = False
        
        # main modules
        self.model = model
        self.discriminator = Discriminator(latent_dim=self.params.slot_size)

        # steps
        self.total_steps = self.params.max_epochs * len(self.datamodule.train_dataloader())
        self.wakeup_sparse_steps = self.total_steps * self.params.wakeup_sparse_mask_pct
        self.automatic_optimization = False

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # preprocess for batch
        if self.params.clevr_with_mask:
            batch, _ = batch["image"], batch["mask"]
        batch_1, batch_2 = batch.split(batch.size(0)//2)
        
        # init optimizers and schedulers
        opt_sa, opt_d = self.optimizers()
        sh_sa = self.lr_schedulers()
        
        # lazy wakeup to activate sparse mask
        self.activate_mask = True if self.global_step > self.wakeup_sparse_steps else False
        
        # First Phase for updating slot encoder-decoder
        recon_combined, _, _, slots, _ = self.model.forward(batch_1, self.activate_mask)
        mse_loss = F.mse_loss(recon_combined, batch_1)
        sparse_loss = torch.mean(torch.abs(F.relu(slots)))
        
        # slots = slots.squeeze()
        # d_slots = self.discriminator(slots)
        # tc_loss = (d_slots[:, 0] - d_slots[:, 1]).mean()
        tc_loss = mse_loss
        
        if self.activate_mask:
            sparse_weight = self.params.sparse_weight if not self.params.auto_sparse_weight \
                else warm_and_decay_annealing(0.001, self.params.sparse_weight, self.global_step - self.wakeup_sparse_steps, self.params.annealing_steps)
        else:
            sparse_weight = 0.0
        anneal_tc_reg = linear_annealing(0, 1, self.global_step, self.params.annealing_steps)
        
        sa_loss = mse_loss + sparse_weight * sparse_loss + \
            anneal_tc_reg * self.params.tc_weight * tc_loss

        # backpropagate loss for slot attention
        opt_sa.zero_grad()
        self.manual_backward(sa_loss, retain_graph=True)
        
        # # Second Phase for updating discriminator
        # _, _, _, slots, _ = self.model.forward(batch_2, self.activate_mask)
        # slots = slots.view(batch_2.size(0), self.params.num_slots, -1)
        # slots_perm = permute_dims(slots).view(batch_2.size(0)*self.params.num_slots, -1).detach()
        # d_slots_perm = self.discriminator(slots_perm)
        
        # ones = torch.ones(batch_2.size(0)*self.params.num_slots, dtype=torch.long, device=batch_2.device)
        # zeros = torch.zeros_like(ones, device=batch_2.device)
        # d_loss = 0.5*(F.cross_entropy(d_slots, zeros) + F.cross_entropy(d_slots_perm, ones))
        d_loss = sa_loss
        
        # # backpropagate loss for discriminator
        # opt_d.zero_grad()
        # self.manual_backward(d_loss)
        
        # update all parameters
        opt_sa.step()
        sh_sa.step()
        # opt_d.step()
        
        logs = {"loss": sa_loss, "d_loss": d_loss, "sparse_weight": sparse_weight, "activate_mask": self.activate_mask}
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

        recon_combined, recons, masks, _, slot_masks = self.model.forward(batch, self.activate_mask)
        recons_vis = torch.zeros_like(recons)
        if slot_masks != None:
            slot_masks = ~slot_masks.eq(0).view(*slot_masks.size(), 1, 1).repeat(1,1, *recons.size()[-2:])
            recons_vis[:,:,0] = recons[:,:,0] * (slot_masks + ~slot_masks)
            recons_vis[:,:,1] = recons[:,:,1] * (slot_masks + ~slot_masks*0.1)
            recons_vis[:,:,2] = recons[:,:,2] * (slot_masks + ~slot_masks*0.1)
        else:
            recons_vis = recons
        # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    batch.unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recons_vis,# raw reconstructions
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
        # preprocess for batch
        if self.params.clevr_with_mask:
            batch, gt_masks = batch["image"], batch["mask"]
        batch_1, batch_2 = batch.split(batch.size(0)//2)
        gt_masks_1, gt_masks_2 = gt_masks.split(batch.size(0)//2)
        
        # First Phase for updating slot encoder-decoder
        recon_combined, _, masks, slots, _ = self.model.forward(batch_1, self.activate_mask)
        
        mse_loss = F.mse_loss(recon_combined, batch_1)
        sparse_loss = torch.mean(torch.abs(F.relu(slots)))
        
        # slots = slots.squeeze()
        # d_slots = self.discriminator(slots)
        # tc_loss = (d_slots[:, 0] - d_slots[:, 1]).mean()
        tc_loss = mse_loss
        
        if self.activate_mask:
            sparse_weight = self.params.sparse_weight if not self.params.auto_sparse_weight \
                else warm_and_decay_annealing(0.001, self.params.sparse_weight, self.global_step - self.wakeup_sparse_steps, self.params.annealing_steps)
        else:
            sparse_weight = 0.0
        anneal_tc_reg = linear_annealing(0, 1, self.global_step, self.params.annealing_steps)
        
        sa_loss = mse_loss + sparse_weight * sparse_loss + \
            anneal_tc_reg * self.params.tc_weight * tc_loss
        
        # Second Phase for updating discriminator
        # _, _, _, slots = self.model.forward(batch_2, self.activate_mask)
        # slots = slots.view(batch_2.size(0), self.params.num_slots, -1)
        # slots_perm = permute_dims(slots).view(batch_2.size(0)*self.params.num_slots, -1).detach()
        # d_slots_perm = self.discriminator(slots_perm)
        
        # ones = torch.ones(batch_2.size(0)*self.params.num_slots, dtype=torch.long, device=batch_2.device)
        # zeros = torch.zeros_like(ones)
        # d_loss = 0.5*(F.cross_entropy(d_slots, zeros) + F.cross_entropy(d_slots_perm, ones))
        d_loss = sa_loss
        
        if self.params.clevr_with_mask:    
            ari = adjusted_rand_index(gt_masks_1, masks, exclude_background=False).mean()
            fgari = adjusted_rand_index(gt_masks_1, masks).mean()
            self.validation_step_outputs.append({"loss": sa_loss.item(), "sparse_loss": sparse_loss.item(), "mse_loss": mse_loss.item(), "tc_loss": tc_loss.item(), "d_loss": d_loss.item(), "ARI": ari.item(), "FG-ARI": fgari.item()})
        else:
            self.validation_step_outputs.append({"loss": sa_loss.item(), "sparse_loss": sparse_loss.item(), "mse_loss": mse_loss.item(), "tc_loss": tc_loss.item(), "d_loss": d_loss.item(),})


    def on_validation_epoch_end(self):
        avg_loss = np.array([x["loss"] for x in self.validation_step_outputs]).mean()
        avg_d_loss = np.array([x["d_loss"] for x in self.validation_step_outputs]).mean()
        avg_sparse_loss = np.array([x["sparse_loss"] for x in self.validation_step_outputs]).mean()
        avg_mse_loss = np.array([x["mse_loss"] for x in self.validation_step_outputs]).mean()
        avg_tc_loss = np.array([x["tc_loss"] for x in self.validation_step_outputs]).mean()
        avg_ari = np.array([x["ARI"] for x in self.validation_step_outputs]).mean()
        avg_fgari = np.array([x["FG-ARI"] for x in self.validation_step_outputs]).mean()
        logs = {
            "avg_val_loss": avg_loss,
            "avg_d_loss": avg_d_loss,
            "avg_sparse_loss": avg_sparse_loss,
            "avg_mse_loss": avg_mse_loss,
            "avg_tc_loss": avg_tc_loss,
            "avg_ari": avg_ari,
            "avr_fgari": avg_fgari,
        }
        self.log_dict(logs, sync_dist=True)
        print("; ".join([f"{k}: {v:.6f}" for k, v in logs.items()]))

    def configure_optimizers(self):
        optimizer_SA = optim.Adam(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.params.lr_d, betas=(0.5,0.9))

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

        scheduler_SA = optim.lr_scheduler.LambdaLR(optimizer=optimizer_SA, lr_lambda=warm_and_decay_lr_scheduler)
        # scheduler_D = optim.lr_scheduler.LambdaLR(optimizer=optimizer_D, lr_lambda=warm_and_decay_lr_scheduler)

        return (
            [optimizer_SA, optimizer_D],
            [scheduler_SA],
        )