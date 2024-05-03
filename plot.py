

import matplotlib.pyplot as plt
import pandas as pd
import os


dir = os.listdir("logs")
dir.sort()
latestfile = "logs/" + dir[-1]

data = pd.read_csv(latestfile, names=["loss"])
plt.yscale("log")
plt.plot(data["loss"])
plt.plot(data["loss"].rolling(200).mean())
plt.show()
