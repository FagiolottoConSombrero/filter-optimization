import pytorch_lightning as pl
import torch.nn.functional as F
from models import *
from filters import *


class LLP(pl.LightningModule):

    def __init__(self, lr=1e-3, num_classes=5, patience=20, model_type=1):
        super().__init__()
        self.model_type = model_type
        self.save_hyperparameters()
        if model_type == 1:
            self.model = SpectralMLP()  # poi clamp nella loss

        self.filter2_module = init_transmittance()  # usa la tua funzione

        self.lr = lr
        self.num_classes = num_classes
        self.patience = patience

    def forward(self, x):  # x = radianza HSI: [B,121,16,16]
        img1, img2 = simulate_two_shots_camera(x, self.filter2_module)
        return self.model(torch.cat((img1, img2), dim=1))  # [B,121,16,16]

    def step(self, batch, stage):
        ref, rad = batch  # ref=[B,121,16,16] riflettanza, rad=[B,121,16,16] radianza

        recon = self(rad)                     # [B,121,16,16], riflettanza ricostruita

        # ---- loss spettrale per riflettanza ----
        loss, mae = spectral_reflectance_loss(recon, ref, lambda_ang=0.2)

        # ---- logging ----
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, batch_size=ref.size(0))
        self.log(f"{stage}_mae", mae,  on_epoch=True, prog_bar=True, batch_size=ref.size(0))

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=self.patience
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_loss"
            }}


def spectral_reflectance_loss(R_pred, R_gt, lambda_ang=0.2, eps=1e-8):
    """
    R_pred, R_gt: [B, 121, H, W]  (riflettanza)
    """
    # clamp per sicurezza (riflettanza fisica 0–1)
    R_pred = torch.clamp(R_pred, 0.0, 1.0)

    # --- L1 su tutta la mappa spettrale ---
    loss_mae = F.l1_loss(R_pred, R_gt)

    # --- termine angolare (SAM-like) per pixel ---
    B, L, H, W = R_pred.shape

    # [B*H*W, L]
    pred_flat = R_pred.permute(0, 2, 3, 1).reshape(-1, L)
    gt_flat = R_gt.permute(0, 2, 3, 1).reshape(-1, L)

    pred_n = pred_flat / (pred_flat.norm(dim=1, keepdim=True) + eps)
    gt_n = gt_flat / (gt_flat.norm(dim=1, keepdim=True) + eps)

    cos_sim = (pred_n * gt_n).sum(dim=1)  # [B*H*W]
    loss_ang = (1.0 - cos_sim).mean()

    return loss_mae + lambda_ang * loss_ang, loss_mae
