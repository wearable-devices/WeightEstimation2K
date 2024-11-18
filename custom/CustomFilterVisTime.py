from training_playground.custom.layers import SEMGScatteringTransform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use(['seaborn-v0_8-whitegrid'])

# Create the transform
semg_transform = SEMGScatteringTransform()

# plot time domain filters
semg_transform.build(input_shape=128)
semg_transform.plot_build_filters()
