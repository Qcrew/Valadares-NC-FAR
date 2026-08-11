# Supplementary Figure 7: coherent-state decay

This directory contains the analysis code and the source data for the coherent-state decay measurements used in Supplementary Figure 7.

| Data directory | Measurement | Retrieved | Coherent-state amplitude | QM DC bias point | YOKO bias | Qubit frequency |
| --- | --- | --- | ---: | --- | --- | --- |
| `data_1` | Try 1: coherent-state decay | 13 July 2026 | $\alpha = 3.9$ | 0 (lower sweet spot) | +34.0 mA | 5.5672 GHz |
| Not included | Try 2: coherent-state decay | 17 July 2026 | $\alpha = 1$ | 0 (lower sweet spot) | +34.0 mA | 5.5672 GHz |
| `data_3` | Try 3: coherent-state decay | 18 July 2026 | $\alpha = 3.36$ | -0.5 (master bias) | +27.495 mA | 5.5672 GHz + 198 MHz = 5.7652 GHz |
| `data_4` | Try 4: coherent-state decay | 21 July 2026 | $\alpha = 3.36$ | -0.75 (master bias) | +24.29 mA | 5.5672 GHz + 398 MHz = 5.9652 GHz |
| `data_5` | Try 5: coherent-state decay | 23 July 2026 | $\alpha = 3.36$ | -0.25 (master bias) | +30.73 mA | 5.5672 GHz + 50.36 MHz = 5.61756 GHz |

`data_1` contains compact per-delay NPZ averages of the raw real and imaginary HDF5 measurement stacks. Recreate it from the original raw data with:

```text
python average_data1.py <raw-data_1-directory> data_1
```

The uploaded data contains only the averaged NPZ products. The Try 2 (`data_2`) raw dataset is intentionally not included. Raw HDF5 stacks, generated figures, Python bytecode, and notebook checkpoints are excluded.
