| Quantity | Value | Meaning |
| --- | --- | --- |
| Domain / duration | [0,1]² / [0,0.8] | Synthetic length/time units |
| u; v; D | 0.35; -0.15; 0.005 | L/T; L/T; L²/T |
| Source | one Gaussian; sigma=0.08 | Q is peak source intensity (C/T) |
| x_s; y_s; Q ranges | [0.25,0.75]; [0.25,0.75]; [0.8,1.4] | IID uniform final design |
| Reference grid / dt | 641² / 0.00003125 | dx=dy=0.0015625; 25600 updates |
| Refinement grids / dt | 161² / 0.0005; 321² / 0.000125 | Richardson estimate, not a bound |
| Inference FD grid / dt | 201² / 0.00025 | Classical and conditional-Q diagnosis |
| Sensors / observations | 20 fixed / 81 times | 0 to 0.8; zero added noise |
| Dense evaluation | 641² nodes at 0.1,0.2,0.4,0.8 | Equal point weights |
| IC / BC | C=0 / C=0 on all four edges | Absorbing Dirichlet; not no-flux |
| Reference accuracy flags | 3/30 retained | Cases 016,017,027 exceed 1% target |
