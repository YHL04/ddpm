

import matplotlib.pyplot as plt


def read_file(filename):
    with open(filename, "r") as f:
        loss = f.readlines()

    return [float(x[:-2]) for x in loss]


loss = read_file("logs/loss.txt")
plt.plot(loss)
plt.show()

