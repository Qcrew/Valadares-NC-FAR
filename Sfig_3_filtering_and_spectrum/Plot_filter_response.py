import numpy as np
import matplotlib.pyplot as plt

# === Configuration ===


filename = "Valadares-Dorogov-FAR/Sfig_3_filtering_and_spectrum/filter_S21.txt"

# === Read data ===
freqs = []
values = []

with open(filename, "r") as f:
    for line in f:
        parts = line.strip().split()
        # Only process lines that look like data (2 numeric columns)
        if len(parts) == 2:
            try:
                freq = float(parts[0])
                val = float(parts[1])
                freqs.append(freq)
                values.append(val)
            except ValueError:
                continue  # skip lines that are not numeric

freqs = np.array(freqs)
values = np.array(values)

# === Plot ===
plt.figure(figsize=(6, 4))
plt.plot(freqs, values)
plt.plot([6.868, 6.868], [-120, 0])
plt.plot([7.634, 7.634], [-120, 0])
plt.plot([5.894, 5.894], [-120, 0])
plt.xlabel("Frequency (GHz)")
plt.ylabel("S-Parameter (dB)")
plt.title("Terminal S Parameter Plot")
plt.grid(True)
plt.tight_layout()
# plt.savefig("s21.pdf", format="pdf", bbox_inches="tight", transparent=True)

plt.show()
