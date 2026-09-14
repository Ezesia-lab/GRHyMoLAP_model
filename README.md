# GRHyMoLAP

A lightweight, calibratable rainfall-runoff model with a scikit-learn-style
API — fit it on precipitation/PET/observed streamflow, then simulate.

```python
from grhymolap import GRHyMoLAP

model = GRHyMoLAP(n_warmup=365)
model.fit(P, PET, Q, dates=dates, train_ratio=0.7)

model.params_        # calibrated (MU, LAMBDA, X1, gamma)
model.train_scores_  # {"nse": ..., "kge": ..., "rmse": ..., ...}
model.val_scores_    # same, on the held-out period

Qsim = model.simulate(P_new, PET_new)  # run on new forcing
```

## Install

```bash
pip install grhymolap
```

(or, for local development: `pip install -e ".[dev]"`)

## Design

- **`grhymolap.model`** — model equations (percolation, routing), JIT-
  compiled with [numba](https://numba.pydata.org/) since these are
  sequential loops re-run on every calibration iteration. No I/O, no
  hidden state; easy to test or wrap elsewhere. First call per process
  pays a one-time compile cost (cached to disk); after that it's
  roughly 50-60x faster than the equivalent pure-Python loop.
- **`grhymolap.metrics`** — `nse`, `nnse`, `kge`, `lognse`, `rmse`, `mae`, `pbias`,
  all NaN-safe.
- **`grhymolap.periods`** — warm-up/train/validation splitting, either
  ratio-based (`train_ratio=0.7`) or explicit date ranges
  (`train_period=(...)`, `val_period=(...)`).
- **`grhymolap.calibration`** — pick an objective (default `"nse"`) and an
  optimizer (default `"nelder-mead"`, also `"l-bfgs-b"` and
  `"differential_evolution"`), or plug in your own via `custom_objective`
  and `custom_optimizer`.
- **`grhymolap.GRHyMoLAP`** — the class most users want: `.fit()` / `.simulate()` / `.score()`.

## Warm-up, training, and validation periods

```python
# Ratio-based (simplest)
model.fit(P, PET, Q, dates=dates, train_ratio=0.7)

# Explicit dates
model.fit(P, PET, Q, dates=dates,
          train_period=("1980-01-01", "2005-12-31"),
          val_period=("2006-01-01", "2014-12-31"))
```

`n_warmup` (set on `GRHyMoLAP(...)`) is always a fixed number of leading
timesteps excluded from scoring in either mode — the model still simulates
through them so its internal state settles before anything is scored.

## Choosing objective and optimizer

```python
model = GRHyMoLAP(objective="kge", optimizer="l-bfgs-b")
```

Available objectives: `"nse"` (default), `"kge"`, `"lognse"`, `"rmse"`, `"mae"`, `"pbias"`.
Available optimizers: `"nelder-mead"` (default), `"l-bfgs-b"`, `"differential_evolution"`.

To use your own objective or optimizer (e.g. a hybrid ML loss, or CMA-ES):

```python
def my_objective(params, Q0, Pn, En, Q_obs, train_mask):
    ...
    return value_to_minimize

def my_optimizer(objective_fn, initial_guesses, bounds, optimizer_kwargs):
    ...
    return best_params

model = GRHyMoLAP(custom_objective=my_objective, custom_optimizer=my_optimizer)
```

## Examples

See `examples/A_CAMELS_FR_basin.py` for a single-basin evaaluation.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).


## Citation
If you use GRHyMoLAP in your research, please cite:

 Houénafa, S. E., Latella, M., Gohouede, L. C., & Sezen, C. (2026). GRHyMoLAP: A process-driven ODE catchment hydrology model inspired by GR4J and HyMoLAP approaches. *Journal of Hydrology*, 135597.
