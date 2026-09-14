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

station ='M334091010'

Q     = dataframe[station].sel(dynamic_features="q_mm_obs").to_numpy()
P     = dataframe[station].sel(dynamic_features="pcp_mm").to_numpy()
PET   = dataframe[station].sel(dynamic_features="pet_mm_pm").to_numpy()


dates = pd.date_range(start="2000-01-01", end="2021-12-31", freq="D")


#Fitting

model = GRHyMoLAP(n_warmup=0, objective="nse", optimizer="nelder-mead")
model.fit(P, PET, Q, dates=dates, train_ratio=0.6)

print("Calibrated params (MU, LAMBDA, X1, gamma):", model.params_)
print("Train scores:", model.train_scores_)
print("Val scores:  ", model.val_scores_)

#Plot
dates = pd.date_range(
    start="2000-01-01",
    periods=len(Q),
    freq="D"
)

Qsim = model.simulate(P, PET)

mask = (dates >= "2020-01-01") & (dates <= "2021-12-31")

plt.figure(figsize=(14, 5))

plt.plot(dates[mask], Q[mask], label="Observed", linewidth=1)
plt.plot(dates[mask], Qsim[mask], label="Simulated", linewidth=1)

plt.xlabel("Date")
plt.ylabel("Streamflow (mm/day)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
