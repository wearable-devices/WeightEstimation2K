from custom.layers import SEMGScatteringTransform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use(['seaborn-v0_8-whitegrid'])

# Create the transform
semg_transform = SEMGScatteringTransform()

# Some data1``
path = '/home/wld-algo-6/DataCollection/Data/Leeor1/Train/Leeor_0_weight_1000_0_Leaning_M.csv'
start = 3000
finish = start + 512
df = pd.read_csv(path)
snc1 = df.Snc1.dropna().to_numpy()
snc1o = snc1
snc1 = snc1[start:finish]
snc2 = df.Snc2.dropna().to_numpy()
snc2o = snc2
snc2 = snc2[start:finish]
snc3 = df.Snc3.dropna().to_numpy()
snc3o = snc3
snc3 = snc3[start:finish]
shape = snc1.shape[0]

# Build defines shape
semg_transform.build(input_shape=128)

# Apply the transform to each signal
Sx1, Sx1p = semg_transform(snc1o)
Sx2, Sx2p = semg_transform(snc2o)
Sx3, Sx3p = semg_transform(snc3o)

plt.subplot(3, 1, 1)
plt.plot(snc1o, label="$Snc_1$")
plt.plot(snc2o, label="$Snc_2$")
plt.plot(snc3o, label="$Snc_3$")
#plt.axvline(x=start, color='r', linestyle='--', label=f'Start ({start})')
#plt.axvline(x=finish, color='r', linestyle='--', label=f'End ({finish})')
plt.xlabel('Samples')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.imshow(np.transpose(np.squeeze(Sx1, axis=0)), aspect='auto', cmap=mpl.colormaps['plasma'])
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
plt.title('Filterbank Energy', loc='left')

plt.subplot(3, 1, 3)
plt.imshow(np.transpose(np.squeeze(Sx1p, axis=0)), aspect='auto', cmap=mpl.colormaps['plasma'])
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
plt.title('Filterbank Energy', loc='left')

plt.autoscale(enable=True, axis='x', tight=True)
plt.legend()
plt.show()
