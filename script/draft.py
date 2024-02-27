import pandas as pd
import numpy as np

dataset = '/Users/sara/PycharmProjects/specparam/data/dataset/LGA/LAMOST_Gaia_APOGEE.csv'
data = pd.read_csv(dataset)

r_med_geo = data['r_med_geo']
r_lo_geo = data['r_lo_geo']
r_hi_geo = data['r_hi_geo']

ag_gspphot = data['ag_gspphot']
ag_gspphot_lower = data['ag_gspphot_lower']
ag_gspphot_upper = data['ag_gspphot_upper']

# absolute G magnitude
# ignore error of apparent G magnitude,
# given the uncertainties on Gaia are genarally less than a few millimagnitudes(0.3-6 mmag for G< 20 mag).
sigma_mg = np.sqrt(
    ((-5 / (r_med_geo * np.log(10))) * ((r_hi_geo - r_lo_geo) / 2)) ** 2 +
    ((ag_gspphot_upper - ag_gspphot_lower) / 2) ** 2
)

print(np.median(sigma_mg), np.mean(sigma_mg))
# 0.017769046694342518 0.027776601939168907

data['sigma_mg'] = sigma_mg
data.to_csv('/Users/sara/PycharmProjects/specparam/data/dataset/LGA/LAMOST_Gaia_APOGEE_new.csv', index=False)