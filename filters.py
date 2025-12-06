import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# ===========================
# 1) Filtro 1 fisso hard-coded
# ===========================

# Carica dati
df = pd.read_csv('/home/acp/Documenti/filters_VIS_IR_resampled_400_1000_5nm.csv')
wavelength = df['wavelength_nm'].values

# Unione
filter1 = df[['transmittance_visible', 'transmittance_infrared']].max(axis=1).values
filter1 = torch.tensor(filter1, dtype=torch.float32)


# ===============================
# 2) Filtro 2 ottimizzabile (sigmoid)
# ===============================

class OptimizedFilter(nn.Module):
    def __init__(self, init_transmittance: torch.Tensor):
        super().__init__()
        self.init_transmittance = init_transmittance
        eps = 1e-6
        x0 = torch.log(self.init_transmittance.clamp(eps, 1 - eps) /
                       (1 - self.init_transmittance.clamp(eps, 1 - eps)))
        self.raw_param = nn.Parameter(x0)  # parametri liberi

    def forward(self):
        # output sempre in [0,1]
        return torch.sigmoid(self.raw_param) # sigmoid serve per tenere i valori nel range 0-1


filter2 = None   # definita una volta, fuori dalle funzioni


def init_transmittance(init_type: int = 2):
    global filter2

    if init_type == 0:
        # inizializzazione complementare
        f2_init = 1.0 - filter1  # [L]
    elif init_type == 1:
        # tutto a 1
        f2_init = torch.ones_like(filter1)
    else:
        # random in [0, 0.2]
        f2_init = torch.rand_like(filter1) * 0.2

    # modulo learnable
    filter2 = OptimizedFilter(f2_init)
    return filter2


def get_sensor_curves(csv_path: str = '/home/acp/Documenti/Sony_ILCE_6100_RGBIR_scaled_005.csv'):
    """
    Legge le curve spettrali del sensore RGB-IR Sony
    e restituisce un tensore PyTorch di shape [4, 121].

    Output:
        S_cam: tensor [4,121] con l'ordine (R, G, B, IR)
    """
    df = pd.read_csv(csv_path)

    # Assumi colonne chiamate così (controlla il CSV):
    R = torch.tensor(df["red"].values, dtype=torch.float32)
    G = torch.tensor(df["green"].values, dtype=torch.float32)
    B = torch.tensor(df["blue"].values, dtype=torch.float32)
    IR = torch.tensor(df["IR850"].values, dtype=torch.float32)

    # --- 2) Nuovo asse a 5 nm ---
    wavelength_5nm = np.arange(400, 1000 + 1, 5)  # 400..1000 → 121 valori
    wavelength = np.arange(400, 1000 + 1, 10)

    # --- 3) Interpolazione ---
    R_5 = np.interp(wavelength_5nm, wavelength, R)
    G_5 = np.interp(wavelength_5nm, wavelength, G)
    B_5 = np.interp(wavelength_5nm, wavelength, B)
    IR_5 = np.interp(wavelength_5nm, wavelength, IR)

    # --- 4) Crea tensore [4,121] ---
    curves = torch.tensor(
        np.stack([R_5, G_5, B_5, IR_5], axis=0),
        dtype=torch.float32
    )

    return curves


def simulate_two_shots_camera(HSI, filter2):
    # HSI:   [B,121,H,W]
    curves = get_sensor_curves().to(HSI.device)  # [4,121]
    f1 = filter1.to(HSI.device)                  # [121] fisso
    f2 = filter2().to(HSI.device)                  # [121] learnable

    eff1 = curves * f1[None, :]   # [4,121]
    eff2 = curves * f2[None, :]   # [4,121]

    img1 = torch.einsum('b l h w, c l -> b c h w', HSI, eff1)
    img2 = torch.einsum('b l h w, c l -> b c h w', HSI, eff2)

    return img1, img2


def simulate_single_shots_camera(HSI):
    # HSI:   [B,121,H,W]
    curves = get_sensor_curves().to(HSI.device)  # [4,121]
    f1 = filter1.to(HSI.device)                  # [121] fisso
    eff1 = curves * f1[None, :]   # [4,121]
    img1 = torch.einsum('b l h w, c l -> b c h w', HSI, eff1)

    return img1






