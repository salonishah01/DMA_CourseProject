import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


""" Idiot check if there are values <0.0 or >1.0 """

submission = pd.read_csv("AttemptX_ghfilter.csv", index_col=0)

print(submission.describe())

print(submission[submission['2008 [YR2008]']>1.0])
print(submission[submission['2012 [YR2012]']>1.0])
print(submission[submission['2008 [YR2008]']<0.0])
print(submission[submission['2012 [YR2012]']<0.0])


fig, ax = plt.subplots(1,2)
col1 = submission['2008 [YR2008]'].tolist()
col2 = submission['2012 [YR2012]'].tolist()
ax[0].hist(col1, bins=50)
ax[1].hist(col2, bins=50)
ax[0].set_xlim(-0.2,1.2)
ax[1].set_xlim(-0.2,1.2)
ax[0].grid(True)
ax[1].grid(True)
ax[0].set_yscale('log')
ax[1].set_yscale('log')

plt.savefig("test",format='png')
