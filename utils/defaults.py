import os
import socket
from pathlib import Path
from munch import munchify

hostname = socket.gethostname()
DEFAULTS = dict()



workspace = Path(__file__).resolve().parents[2]
DEFAULTS['output_root'] = str(workspace / 'Gaussian-Garments/output/ClothTransformer')
DEFAULTS['data_root'] = str(workspace / 'data/GaussianGarments/ClothTransformer/sim_00000/input')
DEFAULTS['aux_root'] = str(workspace / 'data/GaussianGarments/auxiliary')

# Per-experiment paths supplied by run.py, inherited by every native stage.
for key in ('data_root', 'output_root', 'aux_root'):
    if f'GAUGAR_{key.upper()}' in os.environ:
        DEFAULTS[key] = os.environ[f'GAUGAR_{key.upper()}']



DEFAULTS['stage1'] = 'stage1'
DEFAULTS['stage2'] = 'stage2'
DEFAULTS['stage3'] = 'stage3'


DEFAULTS['rgb_images'] = 'rgb_images'
DEFAULTS['garment_masks'] = 'garment_masks'
DEFAULTS['foreground_masks'] = 'foreground_masks'

# turns the dictionary into a Munch object (so you can use e.g. DEFAULTS.data_root)
DEFAULTS = munchify(DEFAULTS)

for d in ['data_root']:
    if not os.path.exists(DEFAULTS[d]):
        raise FileNotFoundError(f"DEFAULTS.{d} ({DEFAULTS[d]}) does not exist! Follow instructions in DataPreparation.md to set it up.")
