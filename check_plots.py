import pickle
import numpy as np
import matplotlib.pyplot as plt

batch_factors = [2, 4, 8, 16, 32][::-1]  # Reverse for better visualization

batch_sizes = []
min_test_CERs = []

for batch_factor in batch_factors:
    # Compute actual batch size
    batch_size = batch_factor
    batch_sizes.append(batch_size)

    # Build model directory name
    modelName = f'katherine_best_stride_len{batch_factor:.4g}'.replace(".", "_")
    dir = '/home/onuralp/Desktop/c243/neural_seq_decoder_project/logs/speech_logs/' + modelName

    # Load training stats
    with open(f"{dir}/trainingStats", "rb") as f:
        tStats = pickle.load(f)

    testCER = tStats["testCER"]
    min_test_CERs.append(np.min(testCER))

    print(f"Min CER: {np.min(testCER)} at stride length {batch_size}")

# ----- Plot Bar Chart -----
plt.figure(figsize=(10, 5))
plt.bar([str(bs) for bs in batch_sizes], min_test_CERs, color='skyblue')

plt.title("Minimum Test PER vs Stride Length")
plt.xlabel("Stride Length")
plt.ylabel("Min Test PER")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig("PER_vs_stride_len.png", dpi=300, bbox_inches='tight')
plt.show()
