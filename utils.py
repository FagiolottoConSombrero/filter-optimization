import pytorch_lightning as pl
import random
from models import *
from filters import *
from dataloader import *
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader


class OptRecon(pl.LightningModule):
    def __init__(self, lr=1e-3, patience=20, model_type=1, in_dim=8):
        super().__init__()
        self.model_type = model_type
        self.save_hyperparameters()
        self.lr = lr
        self.patience = patience
        self.in_dim = in_dim
        self.input = 0

        if in_dim == 8:
            self.filter2_module = init_transmittance()
        if model_type == 1:
            self.model = SpectralMLP(in_dim=self.in_dim)  # poi clamp nella loss
        elif model_type == 2:
            self.model = MST_Plus_Plus(in_channels=self.in_dim)

    def forward(self, x):  # x = radianza HSI: [B,121,16,16]
        if self.in_dim == 8:
            img1, img2 = simulate_two_shots_camera(x, self.filter2_module)
            self.input = torch.cat((img1, img2), dim=1)
        else:
            img1 = simulate_single_shots_camera(x)
            self.input = img1
        return self.model(self.input)  # [B,121,16,16]

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


def make_loaders(data_root, batch_size=8, val_ratio=0.2):
    # carica l'intero dataset
    full_ds = HSIDataset(data_root)
    # generiamo gli indici
    indices = list(range(len(full_ds)))
    train_idx, val_idx = train_test_split(indices, test_size=val_ratio, shuffle=True, random_state=42)
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
    return train_loader, val_loader


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)