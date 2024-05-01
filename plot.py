

import matplotlib.pyplot as plt
import pandas as pd
import os


def read_file(filename):
    with open(filename, "r") as f:
        loss = f.readlines()

    return [float(x[:-2]) for x in loss]


dir = os.listdir("logs")
dir.sort()
latestfile = "logs/" + dir[-2]

data = pd.read_csv(latestfile, names=["loss"])
plt.yscale("log")
plt.plot(data["loss"])
plt.plot(data["loss"].rolling(200).mean())
plt.show()

# loss = read_file("logs/loss.txt")
# loss_mean = [sum(x[:10]) / 10 for x in loss[:-10]]
#
# plt.yscale("log")
#
# plt.plot(loss)
# plt.plot(loss_mean)
#
# plt.show()
#
