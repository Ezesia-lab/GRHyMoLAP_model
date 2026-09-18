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

data_cal = dataframe[station].sel(time=slice("2000-01-01", "2015-12-31"))
data_val = dataframe[station].sel(time=slice("2016-01-01", "2021-12-31"))

Q_cal = data_cal.sel(dynamic_features="q_mm_obs").to_numpy()
P_cal = data_cal.sel(dynamic_features="pcp_mm").to_numpy()
PET_cal = data_cal.sel(dynamic_features="pet_mm_pm").to_numpy()

Q_val = data_val.sel(dynamic_features="q_mm_obs").to_numpy()
P_val = data_val.sel(dynamic_features="pcp_mm").to_numpy()
PET_val = data_val.sel(dynamic_features="pet_mm_pm").to_numpy()

#Calibration
model = GRHyMoLAP(n_warmup=365)

model.fit(P_cal, PET_cal, Q_cal)

print("Calibrated parameters:", model.params_)
print("Calibration scores:", model.calibration_scores_)

#Validation
from grhymolap import nse, kge, lognse, rmse, mae, pbias

Qsim_val = model.simulate(P_val, PET_val)

scores_val = {
    "nse": nse(Q_val, Qsim_val),
    "kge": kge(Q_val, Qsim_val),
    "lognse": lognse(Q_val, Qsim_val),
    "rmse": rmse(Q_val, Qsim_val),
    "mae": mae(Q_val, Qsim_val),
    "pbias": pbias(Q_val, Qsim_val),
}

print("Simulation scores:", scores_val)

#Plot
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
