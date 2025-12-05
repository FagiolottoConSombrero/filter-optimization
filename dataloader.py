import os
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

wavelengths = np.arange(400, 1001, 5)

D65_bb3000 = np.array([
    0.07254245, 0.07905017, 0.08589669, 0.09308118, 0.10060170,
    0.10845529, 0.11663798, 0.12514484, 0.13396999, 0.14310668,
    0.15254733, 0.16228357, 0.17230629, 0.18260571, 0.19317143,
    0.20399249, 0.21505740, 0.22635425, 0.23787069, 0.24959405,
    0.26151137, 0.27360944, 0.28587486, 0.29829410, 0.31085352,
    0.32353945, 0.33633818, 0.34923605, 0.36221948, 0.37527498,
    0.38838918, 0.40154891, 0.41474118, 0.42795321, 0.44117249,
    0.45438675, 0.46758404, 0.48075267, 0.49388130, 0.50695892,
    0.51997486, 0.53291880, 0.54578079, 0.55855124, 0.57122093,
    0.58378105, 0.59622313, 0.60853910, 0.62072127, 0.63276234,
    0.64465537, 0.65639382, 0.66797151, 0.67938261, 0.69062168,
    0.70168363, 0.71256371, 0.72325753, 0.73376103, 0.74407047,
    0.75418244, 0.76409384, 0.77380188, 0.78330406, 0.79259817,
    0.80168228, 0.81055471, 0.81921406, 0.82765918, 0.83588915,
    0.84390328, 0.85170113, 0.85928244, 0.86664716, 0.87379546,
    0.88072768, 0.88744434, 0.89394612, 0.90023388, 0.90630863,
    0.91217153, 0.91782385, 0.92326702, 0.92850259, 0.93353221,
    0.93835767, 0.94298081, 0.94740362, 0.95162815, 0.95565654,
    0.95949101, 0.96313383, 0.96658737, 0.96985404, 0.97293631,
    0.97583669, 0.97855775, 0.98110209, 0.98347236, 0.98567123,
    0.98770139, 0.98956559, 0.99126655, 0.99280704, 0.99418985,
    0.99541775, 0.99649354, 0.99742002, 0.99819999, 0.99883625,
    0.99933160, 0.99968882, 0.99991069, 1.00000000, 0.99995949,
    0.99979192, 0.99950002, 0.99908649, 0.99855403, 0.99790532,
    0.99714301
])


class HSIDataset(Dataset):
    """
    Dataset per LLP su patch HSI 16x16x121.
    Ritorna:
      - ref: riflettanza        [16,16,121]
      - rad: radianza simulata  [16,16,121]
    """
    def __init__(self, root_dir, illuminant, bag_key="data",
                 dtype=torch.float32):
        """
        root_dir:    cartella root che contiene folders 'bags' e 'labels'
        illuminant:  array/tensor [L] con E(λ)
        bag_key:     nome del dataset dentro l'h5
        dtype:       tipo dei tensori per R
        physical:    se True usa L = R * E / pi (Lambertiano)
                     se False usa L = R * E     (più semplice)
        """
        self.root_dir = Path(root_dir)
        self.bag_key = bag_key
        self.dtype = dtype

        # illuminant in tensor [L]
        if isinstance(illuminant, np.ndarray):
            self.E = torch.tensor(illuminant, dtype=dtype)

        self.bag_dir = self.root_dir / "bags"
        self.files = sorted([f for f in os.listdir(self.bag_dir) if f.endswith(".h5")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        # ---- Carica riflettanza R ----
        bag_path = self.bag_dir / fname
        with h5py.File(bag_path, "r") as f:
            ref = f[self.bag_key][...]   # [16,16,121]

        ref = torch.tensor(ref, dtype=self.dtype)   # [H,W,L]
        rad = ref * self.E.view(-1, 1, 1)

        return ref, rad








