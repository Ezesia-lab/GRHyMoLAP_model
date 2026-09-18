# GRHyMoLAP

A lightweight, calibratable rainfall-runoff model with a hydrology-oriented API — calibrate it on precipitation, PET, and observed streamflow, then simulate.

```python
from grhymolap import GRHyMoLAP

model = GRHyMoLAP(n_warmup=365)

model.fit(P, PET, Q)

model.params_              # calibrated (MU, LAMBDA, X1, gamma)
model.calibration_scores_  # {"nse": ..., "kge": ..., "rmse": ..., ...}

Qsim = model.simulate(P_new, PET_new)  # continue the simulation
````

## Install

From GitHub:

```bash
pip install git+https://github.com/Ezesia-lab/GRHyMoLAP_model.git
```

For local development:

```bash
pip install -e ".[dev]"
```

## Design

* **`grhymolap.model`** — model equations (percolation, streamflow), JIT-compiled with [Numba](https://numba.pydata.org/) for the sequential loops repeatedly evaluated during calibration.
* **`grhymolap.metrics`** — `nse`, `nnse`, `kge`, `lognse`, `rmse`, `mae`, `pbias`.
* **`grhymolap.calibration`** — objective and optimizer selection, parameter bounds, initial guesses, and custom calibration functions.
* **`grhymolap.GRHyMoLAP`** — main interface: `.fit()` / `.simulate()` / `.score()`.

## Calibration

Calibration is performed directly on a supplied precipitation, PET, and observed streamflow series:

```python
model.fit(P, PET, Q)
```

The initial streamflow state is automatically set to the first observed streamflow value:

```text
Q0 = Q[0]
```

### Warm-up

`n_warmup` defines the number of timesteps at the beginning of the calibration series that are simulated to allow the model states to settle but excluded from the calibration objective and calibration scores.

```python
model = GRHyMoLAP(n_warmup=365)
```

Set `n_warmup=0` to disable the warm-up.

After calibration, the final model states are retained internally. Subsequent calls to `.simulate()` continue from these states.

## Choosing objective and optimizer

```python
model = GRHyMoLAP(
    objective="kge",
    optimizer="l-bfgs-b",
)
```

Available objectives:

* `"nse"` — default
* `"kge"`
* `"lognse"`
* `"rmse"`
* `"mae"`
* `"pbias"`

Available optimizers:

* `"nelder-mead"` — default
* `"l-bfgs-b"`
* `"differential_evolution"`

## Calibration options

The main calibration settings can be changed when creating the model:

```python
model = GRHyMoLAP(
    n_warmup=365,
    objective="nse",
    optimizer="nelder-mead",
    bounds=[
        (0.5, 3.5),
        (0.01, 300.0),
        (0.0001, 5000.0),
        (0.0, 20.0),
    ],
    initial_guesses=[
        [1.0, 8, 150, 0.1],
        [0.6, 2, 400, 1.0],
        [1.4, 15, 300, 0.5],
        [1.0, 10, 1000, 0.3],
        [1.8, 5, 800, 0.5],
    ],
    optimizer_kwargs={"options": {"maxiter": 2500, "disp": False}},
)
```

The four parameters are ordered as:

```text
(MU, LAMBDA, X1, gamma)
```

Custom parameter bounds and initial guesses can be supplied when needed.

## Custom objective and optimizer

Custom objectives and optimizers are supported:

```python
def my_objective(params, Q0, Pn, En, Q_obs, calibration_mask):
    ...
    return value_to_minimize

def my_optimizer(objective_fn, initial_guesses, bounds, optimizer_kwargs):
    ...
    return best_params

model = GRHyMoLAP(
    custom_objective=my_objective,
    custom_optimizer=my_optimizer,
)
```

## Results

After calibration:

```python
model.params_
model.Q_sim_
model.calibration_scores_
```

Available performance metrics:

* `nse`
* `nnse`
* `kge`
* `lognse`
* `rmse`
* `mae`
* `pbias`

## Simulation

Run the calibrated model on a validation forcing series:

```python
model.fit(P_cal, PET_cal, Q_cal)

Qsim_val = model.simulate(P_val, PET_val)
```

## Scoring

A fitted model can be evaluated on a supplied series:

```python
from grhymolap import nse, kge, lognse, rmse, mae, pbias

scores_val = {
    "nse": nse(Q_val, Qsim_val),
    "kge": kge(Q_val, Qsim_val),
    "lognse": lognse(Q_val, Qsim_val),
    "rmse": rmse(Q_val, Qsim_val),
    "mae": mae(Q_val, Qsim_val),
    "pbias": pbias(Q_val, Qsim_val),
}
```

The scoring method simulates the supplied forcing and calculates the selected performance metric.

## Examples

See `examples/A_CAMELS_FR_basin.py` for a single-basin evaluation.

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
