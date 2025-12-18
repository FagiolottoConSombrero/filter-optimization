import pytorch_lightning as pl
import random
from models import *
from filters import *
from dataloader import *
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader


class OptRecon(pl.LightningModule):
    def __init__(self, lr=1e-3, patience=20, model_type=1, in_dim=8, lambda_ang=0.2):
        super().__init__()
        self.model_type = model_type
        self.save_hyperparameters()
        self.lr = lr
        self.patience = patience
        self.in_dim = in_dim
        self.input = 0
        self.lambda_ang = lambda_ang

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
        #elif self.in_dim == 4:
            #self.input = simulate_single_shots_camera(x)
        else:
            self.input = simulate_no_filter_camera(x)
        return self.model(self.input)  # [B,121,16,16]

    def step(self, batch, stage):
        ref, rad = batch  # ref=[B,121,16,16] riflettanza, rad=[B,121,16,16] radianza

        recon = self(rad)                     # [B,121,16,16], riflettanza ricostruita

        # ---- loss spettrale per riflettanza ----
        loss, mae = spectral_reflectance_loss(recon, ref, self.lambda_ang)

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


def spectral_reflectance_loss(R_pred, R_gt, lambda_ang=0.2, lambda_smooth=0.05, eps=1e-8):
    """
    R_pred, R_gt: [B, L, H, W]
    """
    # clamp fisico
    R_pred = torch.clamp(R_pred, 0.0, 1.0)

    # ---------- MAE spettrale ----------
    loss_mae = F.l1_loss(R_pred, R_gt)

    # ---------- Termine angolare (SAM-like) ----------
    B, L, H, W = R_pred.shape

    pred_flat = R_pred.permute(0, 2, 3, 1).reshape(-1, L)
    gt_flat = R_gt.permute(0, 2, 3, 1).reshape(-1, L)

    pred_n = pred_flat / (pred_flat.norm(dim=1, keepdim=True) + eps)
    gt_n = gt_flat / (gt_flat.norm(dim=1, keepdim=True) + eps)

    cos_sim = (pred_n * gt_n).sum(dim=1)
    loss_ang = (1.0 - cos_sim).mean()

    # ---------- Smoothness spettrale ----------
    loss_smooth = spectral_smoothness_loss(R_pred)

    # ---------- Loss totale ----------
    loss_total = (
        loss_mae
        + lambda_ang * loss_ang
        + lambda_smooth * loss_smooth
    )
    return loss_total


def spectral_smoothness_loss(R):
    # R: [B, L, H, W]
    dr = R[:, 1:, :, :] - R[:, :-1, :, :]
    return dr.abs().mean()


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


def spectral_angle_mapper(s_true: torch.Tensor, s_recon: torch.Tensor) -> torch.Tensor:
    """
    Calcola il SAM per batch.
    s_true, s_recon: (B, C)
    Ritorna: (B,) in radianti
    """
    # prodotto scalare per campione
    dot = (s_true * s_recon).sum(dim=1)           # (B,)

    # norme per campione
    norm_true = torch.norm(s_true, dim=1)         # (B,)
    norm_recon = torch.norm(s_recon, dim=1)       # (B,)

    # coseno dell'angolo
    cosang = dot / (norm_true * norm_recon + 1e-8)
    cosang = cosang.clamp(-1 + 1e-7, 1 - 1e-7)

    angle = torch.acos(cosang)                    # (B,)
    return angle