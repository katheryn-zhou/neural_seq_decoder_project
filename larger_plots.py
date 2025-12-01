import pickle
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameter grid
beamWidths = [1, 3, 5, 7]
smoothings = [0, 0.2, 0.4, 0.6, 0.8]

# Matrix to store min CER for each combination
cer_matrix = np.zeros((len(beamWidths), len(smoothings)))

base_dir = "/home/onuralp/Desktop/c243/neural_seq_decoder_project/logs/speech_logs/"

for i, bw in enumerate(beamWidths):
    for j, sm in enumerate(smoothings):
        
        modelName = f"katherine_best_torch_beamwidth{bw}_smoothing{sm}"
        dir = base_dir + modelName
        
        try:
            with open(f"{dir}/trainingStats", "rb") as f:
                tStats = pickle.load(f)

            minCER = np.min(tStats["testCER"])
            cer_matrix[i, j] = minCER

            print(f"beamWidth={bw}, smoothing={sm} → Min CER = {minCER}")
        
        except FileNotFoundError:
            print(f"[WARNING] Missing: {dir}/trainingStats")
            cer_matrix[i, j] = np.nan

plt.figure(figsize=(10, 6))
plt.imshow(cer_matrix, cmap="viridis", aspect="auto")

plt.colorbar(label="Min Test CER")
plt.xticks(ticks=range(len(smoothings)), labels=smoothings)
plt.yticks(ticks=range(len(beamWidths)), labels=beamWidths)

plt.xlabel("CTC Smoothing")
plt.ylabel("Beam Width")
plt.title("Minimum Test CER vs Beam Width & Smoothing")

for i in range(len(beamWidths)):
    for j in range(len(smoothings)):
        val = cer_matrix[i, j]
        if not np.isnan(val):
            plt.text(j, i, f"{val:.3f}", ha="center", va="center", color="white")

plt.tight_layout()
plt.savefig("torch_CER_beamwidth_smoothing_heatmap.png", dpi=300)
plt.show()
