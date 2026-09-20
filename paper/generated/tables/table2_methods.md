| Setting | B_revised | J1 | Classical |
| --- | --- | --- | --- |
| Field | 3×64 tanh MLP | same MLP | 201² explicit FD |
| Constraints | hard nonnegative, zero IC/BC | same | nonnegative monotone scheme, zero IC/BC |
| Location | sigmoid; Adam | sigmoid; Adam | bounded nonlinear least squares |
| Q | softplus; Adam | analytical PDE profile | analytical observation profile |
| Budget | 12000 Adam updates | 12000 + terminal Q profile | 3 starts; max_nfev=60 each |
| Step size / seed | 0.001 / 42 | 0.001 / 42 | deterministic starts from observations |
| Batch sizes | data/PDE=512; IC=128; BC=64/edge | same; terminal profile=8192 | all 1620 observations |
| Loss weights | data=10; PDE=IC=BC=1, scaled | same | sensor squared mismatch |
| Inference data | identical sparse observations | identical sparse observations | identical sparse observations |
