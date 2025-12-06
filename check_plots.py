import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

batch_factors = [0, 2, 4, 8, 16]#[2, 4, 8, 16, 32]#[4, 8, 16, 32, 64, 128, 256] 
fileName = [0, 2, 4, 8, 16]#[2, 4, 8, 16, 32] #[1/16,1/8,1/4,1/2, 1, 2, 4]

batch_sizes = []
min_test_CERs = []

for i, batch_factor in enumerate(batch_factors):
    batch_size = batch_factor
    batch_sizes.append(batch_size)

    modelName = f'katherine_best_mask_num{fileName[i]:.4g}'.replace(".", "_")
    dir = '/home/onuralp/Desktop/c243/neural_seq_decoder_project/logs/speech_logs/' + modelName

    with open(f"{dir}/trainingStats", "rb") as f:
        tStats = pickle.load(f)

    testCER = tStats["testCER"]
    min_test_CERs.append(np.min(testCER))

    print(f"Min PER: {np.min(testCER)} at stride length {batch_size}")

# ----- Plot Bar Chart -----
plt.figure(figsize=(10, 5))
bars = plt.bar([str(bs) for bs in batch_sizes], min_test_CERs, color='skyblue')
plt.ylim(0, max(min_test_CERs) * 1.15)
plt.title("Minimum Validation PER vs Number of Masks")
plt.xlabel("Number of Masks")
plt.ylabel("Min Validation PER")
plt.grid(axis='y', linestyle='--', alpha=0.6)

# --- Add value labels on top of bars ---
for bar, value in zip(bars, min_test_CERs):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # center of bar
        height,                              # top of bar
        f"{value:.3f}",                       # label format
        ha='center', va='bottom'
    )
plt.tight_layout()
plt.savefig("PER_vs_mask_num_final.png", dpi=300, bbox_inches='tight')
plt.show()
