import numpy as np
import mne
from mne.datasets import sample

data_path = sample.data_path()
fname_cov = data_path / "MEG" / "sample" / "sample_audvis-cov.fif"
fname_evo = data_path / "MEG" / "sample" / "sample_audvis-ave.fif"
