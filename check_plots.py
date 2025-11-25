import pickle
import numpy as np
import matplotlib.pyplot as plt

batch_factors = [1/16, 1/8, 1/4, 1/2, 1, 2, 4][::-1]  # Reverse for better visualization

batch_sizes = []
min_test_CERs = []

for batch_factor in batch_factors:
    # Compute actual batch size
    batch_size = int(batch_factor * 64)
    batch_sizes.append(batch_size)

    # Build model directory name
    modelName = f'speechBaseline4_batchFactor{batch_factor:.4g}'.replace(".", "_")
    dir = '/home/onuralp/Desktop/c243/neural_seq_decoder/logs/speech_logs/' + modelName

    # Load training stats
    with open(f"{dir}/trainingStats", "rb") as f:
        tStats = pickle.load(f)

    testCER = tStats["testCER"]
    min_test_CERs.append(np.min(testCER))

    print(f"Min CER: {np.min(testCER)} at batch size {batch_size}")

# ----- Plot Bar Chart -----
plt.figure(figsize=(10, 5))
plt.bar([str(bs) for bs in batch_sizes], min_test_CERs, color='skyblue')

plt.title("Minimum Test PER vs Batch Size")
plt.xlabel("Batch Size")
plt.ylabel("Min Test PER")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig("PER_vs_batch_size.png", dpi=300, bbox_inches='tight')
plt.show()
