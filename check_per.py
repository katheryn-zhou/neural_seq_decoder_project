import pickle
import numpy as np
import matplotlib.pyplot as plt

min_test_CERs = []
# Build model directory name
modelName = f'katherine_best_causalGaussian'
dir = '/home/onuralp/Desktop/c243/neural_seq_decoder_project/logs/speech_logs/' + modelName

# Load training stats
with open(f"{dir}/trainingStats", "rb") as f:
    tStats = pickle.load(f)

testCER = tStats["testCER"]
min_test_CERs.append(np.min(testCER))
print(f"Min CER: {np.min(testCER)}")
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- 1. Dirac delta input ---
x = torch.zeros(20)
x[5] = 1.0
x = x.view(1, 1, -1)  # (batch, channels, length)

# --- 2. Shift-free causal Gaussian kernel ---
kernel_size = 20
sigma = 5.0

# full symmetric Gaussian
t = torch.arange(kernel_size, dtype=torch.float32)
center = (kernel_size - 1) / 2
kernel = torch.exp(-0.5 * ((t - center) / sigma) ** 2)

# create causal kernel: keep center->end, flip so last element is current time
causal_kernel = kernel[int(center):].flip(0)
causal_kernel = causal_kernel / causal_kernel.sum()  # normalize

weight = causal_kernel.view(1, 1, -1)

# --- 3. Left-pad input for causal convolution ---
pad = len(causal_kernel) - 1
x_padded = F.pad(x, (pad, 0))

# --- 4. Convolve ---
y = F.conv1d(x_padded, weight=weight, groups=1, padding=0)

# --- 5. Plot input, kernel, and output ---
plt.figure(figsize=(10,3))

plt.subplot(1,3,1)
plt.stem(x.flatten())
plt.title("Input Dirac Delta")

plt.subplot(1,3,2)
plt.stem(causal_kernel)
plt.title("Shift-free Causal Kernel")

plt.subplot(1,3,3)
plt.stem(y.flatten().detach())
plt.title("Output after Causal Convolution")

plt.tight_layout()
plt.show()
"""