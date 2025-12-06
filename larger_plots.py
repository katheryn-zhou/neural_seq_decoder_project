import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
# Hyperparameter grid
beamWidths = [1, 3, 5, 7]
smoothings = [0, 0.2, 0.4, 0.6, 0.8]

# Matrix to store min CER for each combination
cer_matrix = np.zeros((len(beamWidths), len(smoothings)))

base_dir = "/home/onuralp/Desktop/c243/neural_seq_decoder_project/logs/speech_logs/"

for i, bw in enumerate(beamWidths):
    for j, sm in enumerate(smoothings):
        
        modelName = f"katherine_best_beamwidth{bw}_smoothing{sm}"
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
plt.imshow(cer_matrix, cmap="gray_r", aspect="auto")

plt.colorbar(label="Min Validation PER")
plt.xticks(ticks=range(len(smoothings)), labels=smoothings)
plt.yticks(ticks=range(len(beamWidths)), labels=beamWidths)

plt.xlabel("CTC Smoothing")
plt.ylabel("Beam Width")
plt.title("Minimum Validation PER vs Beam Width & Smoothing")

# Compute mean while ignoring NaNs
mean_val = np.nanmean(cer_matrix)

for i in range(len(beamWidths)):     # rows
    for j in range(len(smoothings)): # columns
        val = cer_matrix[i, j]
        if not np.isnan(val):
            plt.text(
                j, i,
                f"{val:.3f}",
                ha="center",
                va="center",
                color="white" if val > mean_val else "black"
            )


plt.tight_layout()
plt.savefig("PER_beamwidth_smoothing_heatmap_final.png", dpi=300)
plt.show()
