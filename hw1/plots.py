import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_theme()

def plot_q2(mean, std, bc, expert, title, ax, x=None, x_label='Iteration'):
    if x is None:
        x = np.arange(len(mean)) + 1
    df = pd.DataFrame.from_dict({'BC': mean, 'std': std, x_label: x, 'BC (1000)':[bc] * len(mean), 'Expert':[expert]*len(mean)})
    ax = df.plot(x=x_label, y='BC', yerr=std, ax=ax)
    df.plot(x=x_label, y='BC (1000)', ax=ax, style='--')
    df.plot(x=x_label, y='Expert',ax=ax, style='--')
    ax.set_title(title, fontsize=20)
    ax.set_ylabel('Return')

# Q1 3)
mean = [342.5796203613281, 3458.939208984375, 3472.720947265625, 3915.314453125, 3122.467529296875]
std = [545.9176025390625, 2262.548095703125, 2519.89306640625, 2289.611328125, 2595.910888671875]
x = [2000, 4000, 6000, 8000, 10000]
bc =2508.73486328125
expert = 5566.845703125
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

plot_q2(mean, std, bc, expert, title='Increasing Train Steps', ax=ax, x=x, x_label='Train Steps')
plt.tight_layout()
plt.savefig('train_steps.pdf')
plt.show()



# Ant
mean = [4091.913330078125, 4824.1748046875, 4807.97119140625, 4558.79345703125, 4320.77587890625, 4756.50048828125,  4730.2666015625, 4709.78955078125,  4754.73828125, 4636.12353515625]
std = [1074.8009033203125, 69.79945373535156, 141.34078979492188, 314.642578125,  1190.9481201171875, 107.51022338867188, 83.43251037597656, 109.5824966430664, 53.023216247558594, 564.796875]
bc = 4091.913330078125
expert = 4713.6533203125

fig, axes = plt.subplots(2, 1, figsize=(8, 10))

plot_q2(mean, std, bc, expert, 'Ant', axes[0])


# 2D Walker
mean = [2508.73486328125, 3424.60986328125,1902.1282958984375, 2527.189208984375, 5002.1328125, 4878.77880859375, 3141.737060546875, 4399.89013671875, 3855.489990234375, 4105.326171875]
std = [ 1298.757080078125, 1698.049560546875, 2194.502685546875, 1377.5965576171875, 1330.1876220703125, 1388.468994140625, 2514.384521484375, 1891.2080078125,  2264.787841796875, 2101.323974609375]
bc =2508.73486328125
expert = 5566.845703125
plot_q2(mean, std, bc, expert, 'Walker 2D', axes[1])

plt.tight_layout()
plt.savefig('dagger.pdf')
plt.show()
