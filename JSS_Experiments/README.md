## Difference between run.py and run-fast.py

Due to the long execution time of `run.py`, we provide another Python file: `run-fast.py`.
`run-fast.py` uses the pre-trained model and only runs PSC-related experiments provided in `run.py`, which means it does not run the k-means and spectral clustering parts from `run.py`. If the user wants to see the full experimental results, please run `run.py`. However, if the user only wants to see the performance and results of parametric spectral clustering, then `run-fast.py` may be more efficient.

## How to run experiments

for run.py:

```sh
cd JSS_Experiments
python run.py
```

for run-fast.py:

```sh
cd JSS_Experiments
python run-fast.py
```
