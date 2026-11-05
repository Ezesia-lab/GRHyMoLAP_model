# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from scipy.optimize import minimize # USE IN THE MODEL CALIBRATION

#import zipfile
#import os

import warnings
warnings.filterwarnings('ignore')

"""##CAMELS-DATA"""

pip install aqua-fetch

import aqua_fetch
print(aqua_fetch.__file__)

from aqua_fetch import RainfallRunoff

rr = RainfallRunoff("CAMELS_FR")

meta, dataframe = rr.fetch()

import xarray as xr

print(dataframe)              # global overview

print(dataframe.dynamic_features.values)

# Full period
print(dataframe["time"].values[0], dataframe["time"].values[-1])

station = 'H050301001'

#Data
data_cal = dataframe[station].sel(time=slice("2000-01-01", "2015-12-31"))
data_val = dataframe[station].sel(time=slice("2016-01-01", "2021-12-31"))

Q_cal = data_cal.sel(dynamic_features="q_mm_obs").to_numpy()
P_cal = data_cal.sel(dynamic_features="pcp_mm").to_numpy()
PET_cal = data_cal.sel(dynamic_features="pet_mm_pm").to_numpy()

Q_val = data_val.sel(dynamic_features="q_mm_obs").to_numpy()
P_val = data_val.sel(dynamic_features="pcp_mm").to_numpy()
PET_val = data_val.sel(dynamic_features="pet_mm_pm").to_numpy()


"""#GRHyMoLAP"""

!pip install git+https://github.com/Ezesia-lab/GRHyMoLAP_model.git #Please compile this in a separate cell

from grhymolap import simulate_streamflow
from grhymolap import nse, kge, rmse, pbias

# ============================================
# Calibration
# ============================================

n_warmup = 365


# Objective function defined by the user
def objective(params, P, E, Q_obs, Q0, n_warmup):
    Qsim = simulate_streamflow(params, Q0, P, E)

    Qsim = Qsim[n_warmup:]
    Q_obs = Q_obs[n_warmup:]

    return 1 - nse(Q_obs, Qsim)


# User-defined parameter bounds
bounds = [
    (0.5, 4.0),       # MU
    (1e-6, 300.0),    # LAMBDA
    (1e-6, 5000.0),   # X1
    (0.0, 20.0)       # gamma
]

# User-defined initial guesses
initial_guesses = [
    [1.0, 8, 150, 0.1],
    [0.6, 2, 400, 1],
    [1.4, 15, 300, 0.5],
    [1.0, 10, 1000, 0.3],
    [1.8, 5, 800, 0.5],
]

# Optimization
results = []

for guess in initial_guesses:
    res = minimize(
        objective,
        guess,
        args=(P_cal, PET_cal, Q_cal, Q_cal[0], n_warmup),
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": 2500, "disp": False}
    )
    results.append(res)

# Select the best optimization result
best_result = min(results, key=lambda res: res.fun)
best_params = best_result.x

print("Calibrated parameters:", best_params)

# Calibration simulation and scores
Qsim_cal = simulate_streamflow(best_params, Q_cal[0], P_cal, PET_cal)
Qsim_cal = Qsim_cal[n_warmup:]
Qobs_cal = Q_cal[n_warmup:]

scores_cal = {
    "nse": nse(Qobs_cal, Qsim_cal),
    "kge": kge(Qobs_cal, Qsim_cal),
    "rmse": rmse(Qobs_cal, Qsim_cal),
    "pbias": pbias(Qobs_cal, Qsim_cal),
}

print("Calibration scores:", scores_cal)

# ============================================
# Validation
# ============================================

# Concatenate calibration and validation periods
P_all = np.concatenate([P_cal, P_val])
PET_all = np.concatenate([PET_cal, PET_val])

# Simulate
Qsim_all = simulate_streamflow(best_params, Q_cal[0], P_all, PET_all)

# Extract the validation period
Qsim_val = Qsim_all[len(Q_cal):]

scores_val = {
    "nse": nse(Q_val, Qsim_val),
    "kge": kge(Q_val, Qsim_val),
    "rmse": rmse(Q_val, Qsim_val),
    "pbias": pbias(Q_val, Qsim_val),
}

print("Validation scores:", scores_val)

"""#Plot"""

dates_val = data_val.time.to_numpy()

plt.figure(figsize=(14, 5))

plt.plot(dates_val, Q_val, label="Observed", linewidth=1)
plt.plot(dates_val, Qsim_val, label="Simulated", linewidth=1)

plt.xlabel("Date")
plt.ylabel("Streamflow (mm/day)")

plt.text(
    0.22,
    0.95,
    f"Station: H050301001\nNSE_val = {nse(Q_val, Qsim_val):.3f}\nKGE_val = {kge(Q_val, Qsim_val):.3f}",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.8)
)

plt.title("Validation")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
