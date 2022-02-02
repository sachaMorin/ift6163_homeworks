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
mean = [2611.607666015625, 342.5796203613281, 3458.939208984375, 3472.720947265625, 3915.314453125, 3122.467529296875]
std = [1187.61669921875, 545.9176025390625, 2262.548095703125, 2519.89306640625, 2289.611328125, 2595.910888671875]
x = [1000, 2000, 4000, 6000, 8000, 10000]
bc = 2611.607666015625
expert = 5566.845703125
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

plot_q2(mean, std, bc, expert, title='Increasing Train Steps', ax=ax, x=x, x_label='Train Steps')
plt.tight_layout()
plt.savefig('train_steps.pdf')
plt.show()



# Ant
mean = [3719.77294921875, 4781.24072265625, 4648.576171875, 4635.1298828125, 4508.0849609375, 4376.9228515625,
        4360.7333984375, 4651.20361328125, 4642.46240234375, 4686.474609375]
std = [1431.041259765625, 117.50680541992188, 152.31980895996094, 111.0687255859375, 101.54996490478516,
       1283.3131103515625, 1155.0489501953125, 90.18067169189453, 107.71580505371094, 97.27669525146484]
bc = 3719.77294921875
expert = 4713.6533203125

fig, axes = plt.subplots(2, 1, figsize=(8, 10))

plot_q2(mean, std, bc, expert, 'Ant', axes[0])


# 2D Walker
mean = [2611.607666015625, 440.1575012207031, 2414.2177734375,  5130.08837890625, 4191.8408203125, 1204.24658203125, 4620.53955078125, 5314.67138671875, 5178.435546875, 5437.4697265625]
std = [1187.61669921875, 1165.442626953125, 1909.4150390625, 859.1868286132812, 2188.655029296875, 1967.8050537109375, 1651.4088134765625, 281.9724426269531,  994.2650146484375, 65.40522766113281]
bc =2611.607666015625
exper = 5566.845703125
plot_q2(mean, std, bc, expert, 'Walker 2D', axes[1])

plt.tight_layout()
plt.savefig('dagger.pdf')
plt.show()
