import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps
import itertools
import seaborn as sns
sns.set(style='ticks', rc={"grid.linewidth": 0.1})
sns.set_context("paper", font_scale=2.5)
color = sns.color_palette("Set2", 6)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# shape (alpha)
# scale (1/beta)
# Mean : E(X) = alpha / beta
# Variance : var(X) = alpha / beta ^ 2
# Moment generating function (MGF): MX(t) = 1 / (1 - beta * t) ^ alpha

# a change in beta will show a sharp change

def plot_gamma(shape, scale, color):

    s = np.random.gamma(shape, scale, 10000)

    count, bins, ignored = plt.hist(s, 50, density=True, color=color, alpha=0.3)

    y = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))

    plt.plot(bins, y, linewidth=2, color=color, label= r"$\alpha: $" + str(shape) + r"$ - \beta: $" + str(1/scale))

def plot_cdf(shape, scale, color):

    s = np.random.gamma(shape, scale, 10000)

    count, bins, ignored = plt.hist(s, 50, density=True, color=color, alpha=0.3, cumulative=True)

    plt.plot(bins[1:], count, linewidth=2, color=color, label= r"$\alpha: $" + str(shape) + r"$ - \beta: $" + str(1/scale))


alpha = [1, 2, 3]
beta = [1, 2, 3]

number = len(alpha) * len(beta)
cmap = plt.get_cmap("jet")
colors = cmap(np.linspace(0, 1, number))
colors = dict(zip(np.arange(number), colors))

combinations = []
i = 0

for alpha, beta in itertools.product(alpha, beta):

    comb = {}
    comb["shape"] = alpha
    comb["scale"] = 1 / beta
    comb["color"] = colors[i]
    combinations.append(comb)
    i += 1

plt.figure(figsize=(10, 8))

for comb in combinations:

    plot_gamma(**comb)

plt.xlabel("precision")
plt.ylabel("probability density")
plt.xlim(0, 4)
# plt.title("PDF of Gamma Distribution")
plt.legend(loc=1, frameon=True, fancybox=False, ncol=3, framealpha=0.5, edgecolor="black")

plt.figure(figsize=(10, 8))

for comb in combinations:

    plot_cdf(**comb)

plt.xlabel("precision")
plt.ylabel("cumulative probability density")
plt.xlim(0, 4)
# plt.ylim(0, 1)
# plt.title("CDF of Gamma Distribution")
plt.legend(loc=4, frameon=True, fancybox=False, ncol=2, framealpha=0.5, edgecolor="black")
plt.show()