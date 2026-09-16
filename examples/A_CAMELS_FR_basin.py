## Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from scipy.optimize import minimize # USE IN THE MODEL CALIBRATION
import warnings
warnings.filterwarnings('ignore')

## CAMELS-DATA : Got through aqua-fetch library
pip install aqua-fetch  #Please, run this alone in a single cell above.

import aqua_fetch
print(aqua_fetch.__file__)

from aqua_fetch import RainfallRunoff

rr = RainfallRunoff("CAMELS_FR")

meta, dataframe = rr.fetch()

## GRHyMoLAP model
!pip install git+https://github.com/Ezesia-lab/grhymolap.git

from grhymolap import GRHyMoLAP

#Station and data
station = 'H050301001'

data = dataframe[station].sel(time=slice("2000-01-01", "2021-12-31"))

Q = data.sel(dynamic_features="q_mm_obs").to_numpy()
P = data.sel(dynamic_features="pcp_mm").to_numpy()
PET = data.sel(dynamic_features="pet_mm_pm").to_numpy()

dates = data.time.to_numpy()


#Fitting
model = GRHyMoLAP(
    n_warmup=0,
    objective="nse",
    optimizer="nelder-mead",
    optimizer_kwargs={"options": {"maxiter": 2500, "disp": False}},
)

model.fit(
    P,
    PET,
    Q,
    dates=dates,
    calibration_period=("2000-01-01", "2012-12-31"),
    validation_period=("2013-01-01", "2021-12-31"),
)

print("Calibrated params (MU, LAMBDA, X1, gamma):", model.params_)

print("Calibration scores:", model.calibration_scores_)

print("Validation scores:  ", model.validation_scores_)


#Plot
dates = pd.date_range(
    start="2000-01-01",
    periods=len(Q),
    freq="D"
)

Qsim = model.simulate(P, PET)

mask = (dates >= "2020-01-01") & (dates <= "2021-12-31")

nse_val = model.validation_scores_["nse"]
kge_val = model.validation_scores_["kge"]

plt.figure(figsize=(14, 5))

plt.plot(dates[mask], Q[mask], label="Observed", linewidth=1)
plt.plot(dates[mask], Qsim[mask], label="Simulated", linewidth=1)

plt.xlabel("Date")
plt.ylabel("Streamflow (mm/day)")

plt.text(
    0.22,
    0.95,
    f"Station: H050301001\nNSE_val = {nse_val:.3f}\nKGE_val = {kge_val:.3f}",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.8)
)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
