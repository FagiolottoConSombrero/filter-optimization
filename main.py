from utils import *
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import argparse


def main(
    data_root: str,
    save_dir: str = "runs/llp",
    input_dim: int = 8,
    model_type: int = 1,
    batch_size: int = 8,
    epochs: int = 50,
    seed: int = 42,
    patience_loss: int = 50,
    patience_es: int = 50,
    devices="auto",
    learning_rate: float = 1e-3,
    lambda_ang: float = 0.2
):

    set_seed(seed)

    # ----- dataloader -----
    train_loader, val_loader = make_loaders(
        data_root=data_root,
        batch_size=batch_size,
        val_ratio=0.2
    )

    # ----- modello LLP -----
    model = OptRecon(lr=learning_rate,
                     patience=patience_loss,
                     model_type=model_type,
                     in_dim=input_dim,
                     lambda_ang=lambda_ang)

    # ----- callbacks -----
    ckpt = ModelCheckpoint(dirpath=save_dir, filename="best",monitor="val_loss", mode="min", save_top_k=1)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=patience_es, verbose=True)
    lrmon = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        default_root_dir=save_dir,
        max_epochs=epochs,
        accelerator="auto",
        devices=devices,
        precision="32-true",
        callbacks=[ckpt, early, lrmon],
        log_every_n_steps=10,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


# ---------------- CLI ----------------
if __name__ == "__main__":

    arg = argparse.ArgumentParser()
    arg.add_argument("--data_root", type=str, required=True)
    arg.add_argument("--save_dir", type=str, default="runs/recon")
    arg.add_argument("--batch_size", type=int, default=32)
    arg.add_argument("--input_dim", type=int, default=8)
    arg.add_argument("--model_type", type=int, default=1)
    arg.add_argument("--epochs", type=int, default=5000)
    arg.add_argument("--seed", type=int, default=42)
    arg.add_argument("--patience_loss", type=int, default=50)
    arg.add_argument("--patience_early_stopping", type=int, default=100)
    arg.add_argument("--devices", type=str, default="auto")
    arg.add_argument("--lr", type=float, default=1e-3)
    arg.add_argument("--lambda_ang", type=float, default=0.2)

    args = arg.parse_args()

    main(
        data_root=args.data_root,
        save_dir=args.save_dir,
        input_dim=args.input_dim,
        model_type=args.model_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        patience_loss=args.patience_loss,
        patience_es=args.patience_early_stopping,
        devices=args.devices,
        learning_rate=args.lr,
        lambda_ang=args.lambda_ang
    )