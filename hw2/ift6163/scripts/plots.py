import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme()
from tensorflow.python.summary.summary_iterator import summary_iterator



def plot(train_mean, train_std,  eval_mean, eval_std, title, ax=None, x=None, x_label='Iteration'):
    if x is None:
        x = np.arange(len(train_mean)) + 1
    if ax is None:
        _, ax= plt.subplots(1,1, figsize=(6,6))
        
    df = pd.DataFrame.from_dict({'Train Avg. Return': train_mean, 
                                 'Train Std. Return': train_std,
                                 'Eval. Avg. Return': eval_mean,
                                 'Eval. Std. Return': eval_std,
                                 x_label: x})
    df.plot(x=x_label, y='Train Avg. Return', yerr='Train Std. Return', ax=ax, marker='o')
    df.plot(x=x_label, y='Eval. Avg. Return', yerr='Eval. Std. Return', ax=ax, marker='o')
    ax.set_xticks(x)

    ax.set_title(title, fontsize=20)
    ax.set_ylabel('Return')
    ax.legend()

def read_event(path):
    results = dict()

    # Parse TF log
    for e in summary_iterator(path):
        for v in e.summary.value:
            if v.tag not in results:
                results[v.tag] = []
            results[v.tag].append(v.simple_value)
    return results

def plot_event(path, title, file_name, ax=None):
    # Question 2
    results = read_event(path)
    plot(results['Train_AverageReturn'], results['Train_StdReturn'], results['Eval_AverageReturn'], results['Eval_StdReturn'], title=title, ax=ax)

    if ax is None:
        plt.savefig(os.path.join('plots', file_name + '.pdf'))
        plt.show()

# Q1
path_q1_1 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-11/19-14-33/data/hw2_q1_cheetah_n500_arch1x32_cheetah-ift6163-v0_11-02-2022_19-14-33/events.out.tfevents.1644624873.chu-G15"
path_q1_2 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-11/19-18-36/data/hw2_q1_cheetah_n5_arch2x250_cheetah-ift6163-v0_11-02-2022_19-18-36/events.out.tfevents.1644625116.chu-G15"
path_q1_3 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-11/19-21-08/data/hw2_q1_cheetah_n500_arch2x250_cheetah-ift6163-v0_11-02-2022_19-21-08/events.out.tfevents.1644625268.chu-G15"

# Q2
path_q2 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-11/17-15-06/data/hw2_q2_obstacles_singleiteration_obstacles-ift6163-v0_11-02-2022_17-15-06/events.out.tfevents.1644617706.chu-G15"
plot_event(path_q2, '1-iteration MPC', 'q_2')

# Q3
fig, axes = plt.subplots(3, 1, figsize=(7, 14))
path_q3_1 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-11/18-01-54/data/hw2_q3_obstacles_obstacles-ift6163-v0_11-02-2022_18-01-54/events.out.tfevents.1644620514.chu-G15"
plot_event(path_q3_1, 'Obstacles', 'q_3_1', ax=axes[0])

path_q3_2 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-11/18-08-03/data/hw2_q3_reacher_reacher-ift6163-v0_11-02-2022_18-08-03/events.out.tfevents.1644620883.chu-G15"
plot_event(path_q3_2, 'Reacher', 'q_3_2', ax=axes[1])

path_q3_3 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-11/18-31-22/data/hw2_q3_cheetah_cheetah-ift6163-v0_11-02-2022_18-31-23/events.out.tfevents.1644622283.chu-G15"
plot_event(path_q3_3, 'Cheetah', 'q_3_3', ax=axes[2])
fig.tight_layout()
plt.savefig(os.path.join('plots', 'q_3.pdf'))
plt.show()

