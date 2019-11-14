import pandas as pd
import numpy as np 
from sklearn import linear_model

clf = linear_model.Lasso(alpha=0.1)

exp_path = "clean-db/"
imp_path = "dbs/"

df2 = pd.read_csv(exp_path + "admissions_predict_LOS2.csv")

dfX = df2[df2.columns[16:25]]
dfY = df2["LOS"]

print dfX
print dfY

clf.fit(dfX, dfY)
print clf.coef_
print list(dfX)
