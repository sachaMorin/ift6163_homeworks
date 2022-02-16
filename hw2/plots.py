import shutil
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme()
from tensorflow.python.summary.summary_iterator import summary_iterator


def get_event_path(dir_path):
    """This is a hack to auto detect the unique file starting with event in the target path."""
    logdir = os.path.join(dir_path, 'event*')
    event_path = glob.glob(logdir)[0]
    return event_path


def read_event(dir_path):
    """Read stupid tensorflow event file. Return a dict where the metric name is the key and the attached list contains the values"""
    # This is a hack to auto detect the unique file starting with event in the target path
    event_path = get_event_path(dir_path)
    results = dict()

    # Parse TF log
    for e in summary_iterator(event_path):
        for v in e.summary.value:
            if v.tag not in results:
                results[v.tag] = []
            results[v.tag].append(v.simple_value)
    return results


def plot(train_mean, train_std, eval_mean, eval_std, title, ax=None, x=None, x_label='Iteration'):
    """Standard mean + std plot for Eval and Train over iterations"""
    if x is None:
        x = np.arange(len(train_mean)) + 1
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

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


def plot_event(path, title, ax=None):
    """Single performance plot over iterations. Takes in the directory with the deisred path."""
    results = read_event(path)
    plot(results['Train_AverageReturn'], results['Train_StdReturn'], results['Eval_AverageReturn'],
         results['Eval_StdReturn'], title=title, ax=ax)


def plot_hyperparams(x, path_list, title, x_label, ax=None):
    """Plot eval and train average returns + std over multiple events testing different hyperparameters
    The max avg return (and the associated standard dev) is retained for each event"""
    results_hyperparam = {'Train_AverageReturn': list(), 'Train_StdReturn': list(), 'Eval_AverageReturn': list(),
                          'Eval_StdReturn': list()}
    for path in path_list:
        results = read_event(path)
        for s in ['Eval', 'Train']:
            avg_return_i = np.array(results[s + '_AverageReturn'])
            results_hyperparam[s + '_AverageReturn'].append(avg_return_i.max())
            results_hyperparam[s + '_StdReturn'].append(results[s + '_StdReturn'][avg_return_i.argmax()])

    plot(results_hyperparam['Train_AverageReturn'], results_hyperparam['Train_StdReturn'],
         results_hyperparam['Eval_AverageReturn'],
         results_hyperparam['Eval_StdReturn'], title=title, ax=ax, x=x, x_label=x_label)


def plot_eval_multiple(exp_names, path_list, title, ax=None, x=None, x_label=None):
    """Take multiple events and plot to Eval Average return and std accross iterations with different lines"""
    results_all = {}

    for i, path in enumerate(path_list):
        results = read_event(path)
        results_all[exp_names[i]] = results['Eval_AverageReturn']
        results_all[exp_names[i] + "_error"] = results['Eval_StdReturn']

    if x is None:
        x = np.arange(len(results_all[exp_names[0]])) + 1
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
    x_label = x_label if x_label is not None else 'Iteration'
    results_all[x_label] = x

    df = pd.DataFrame.from_dict(results_all)

    for name in exp_names:
        df.plot(x=x_label, y=name, yerr=name + '_error', ax=ax, marker='o')
    ax.set_xticks(x)

    ax.set_title(title, fontsize=20)
    ax.set_ylabel('Return')
    ax.legend()


def collect_events(path_list, target_path='data'):
    """Collect the events from all the paths in path list. Copy them and their parent directory (for identification)
    to a new folder."""
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.mkdir(target_path)

    for path in path_list:
        parent = os.path.split(path)[-1]
        os.mkdir(os.path.join(target_path, parent))
        event_path = get_event_path(path)
        file_name = os.path.basename(event_path)
        shutil.copy(event_path, os.path.join(target_path, parent, file_name))

    # shutil.make_archive(base_name=target_path, root_dir=target_path, format='zip')


# Driver code form Hw2
if __name__ == '__main__':
    # Paths (only need the directory where the event file is saved)
    # Q1
    path_q1_1 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-11/19-14-33/data/hw2_q1_cheetah_n500_arch1x32_cheetah-ift6163-v0_11-02-2022_19-14-33"
    path_q1_2 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-11/19-18-36/data/hw2_q1_cheetah_n5_arch2x250_cheetah-ift6163-v0_11-02-2022_19-18-36"
    path_q1_3 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-11/19-21-08/data/hw2_q1_cheetah_n500_arch2x250_cheetah-ift6163-v0_11-02-2022_19-21-08"

    # Q2
    path_q2 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-11/17-15-06/data/hw2_q2_obstacles_singleiteration_obstacles-ift6163-v0_11-02-2022_17-15-06"

    # Q3
    path_q3_1 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-11/18-01-54/data/hw2_q3_obstacles_obstacles-ift6163-v0_11-02-2022_18-01-54"
    path_q3_2 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-11/18-08-03/data/hw2_q3_reacher_reacher-ift6163-v0_11-02-2022_18-08-03"
    path_q3_3 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-11/18-31-22/data/hw2_q3_cheetah_cheetah-ift6163-v0_11-02-2022_18-31-23"

    # Q4
    path_q4_horizon5 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-12/11-42-17/data/hw2_q4_reacher_horizon5_reacher-ift6163-v0_12-02-2022_11-42-17"
    path_q4_horizon15 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-12/11-49-59/data/hw2_q4_reacher_horizon15_reacher-ift6163-v0_12-02-2022_11-49-59"
    path_q4_horizon30 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-12/12-03-38/data/hw2_q4_reacher_horizon30_reacher-ift6163-v0_12-02-2022_12-03-38"
    path_q4_numseq100 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-12/12-30-24/data/hw2_q4_reacher_numseq100_reacher-ift6163-v0_12-02-2022_12-30-24"
    path_q4_numseq1000 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-12/12-52-24/data/hw2_q4_reacher_numseq1000_reacher-ift6163-v0_12-02-2022_12-52-24"
    path_q4_ensemble1 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-12/14-13-50/data/hw2_q4_reacher_ensemble1_reacher-ift6163-v0_12-02-2022_14-13-50"
    path_q4_ensemble3 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-12/14-21-21/data/hw2_q4_reacher_ensemble3_reacher-ift6163-v0_12-02-2022_14-21-21"
    path_q4_ensemble5 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-12/14-43-11/data/hw2_q4_reacher_ensemble5_reacher-ift6163-v0_12-02-2022_14-43-11"

    # Q5
    path_q5_random = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-12/15-02-54/data/hw2_q5_cheetah_random_cheetah-ift6163-v0_12-02-2022_15-02-54"
    path_q5_cem_2 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-12/15-12-56/data/hw2_q5_cheetah_cem_2_cheetah-ift6163-v0_12-02-2022_15-12-56"
    path_q5_cem_4 = "/home/sacha/Documents/ift6163_homeworks/hw2/outputs/2022-02-12/15-31-36/data/hw2_q5_cheetah_cem_4_cheetah-ift6163-v0_12-02-2022_15-31-36"

    # Plots
    plot_path = "/home/sacha/Documents/ift6163_homeworks/hw2/plots"
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)
    os.mkdir(plot_path)

    # Q2 plot
    plot_event(path_q2, '1-iteration MPC')
    plt.savefig(os.path.join(plot_path, 'q_2.pdf'))
    plt.show()

    # Q3 plots
    fig, axes = plt.subplots(3, 1, figsize=(7, 14))
    plot_event(path_q3_1, 'Obstacles', ax=axes[0])
    plot_event(path_q3_2, 'Reacher', ax=axes[1])
    plot_event(path_q3_3, 'Cheetah', ax=axes[2])
    fig.tight_layout()
    plt.savefig(os.path.join(plot_path, 'q_3.pdf'))
    plt.show()

    # Q4 plots
    fig, axes = plt.subplots(3, 1, figsize=(7, 14))
    plot_eval_multiple(exp_names=['5', '15', '30'], path_list=[path_q4_horizon5, path_q4_horizon15, path_q4_horizon30], title='Planning Horizon', ax=axes[0])
    plot_eval_multiple(exp_names=['100', '1000'], path_list=[path_q4_numseq100, path_q4_numseq1000], title='MPC Number of Action Sequences', ax=axes[1])
    plot_eval_multiple(exp_names=['1', '3', '5'], path_list=[path_q4_ensemble1, path_q4_ensemble3, path_q4_ensemble5], title='Ensemble Size', ax=axes[2])
    fig.tight_layout()
    plt.savefig(os.path.join(plot_path, 'q_4.pdf'))
    plt.show()

    # Q5 Plot
    plot_eval_multiple(exp_names=['Random', 'CEM (2 iter.)', 'CEM (4 iter.)'],
                       path_list=[path_q5_random, path_q5_cem_2, path_q5_cem_4],
                       title='CEM comparison')
    plt.savefig(os.path.join(plot_path, 'q_5.pdf'))
    plt.show()

    # Collect all the event files in a new data directory
    data_path = "/home/sacha/Documents/ift6163_homeworks/hw2/data"
    path_list = [path_q1_1,
                 path_q1_2,
                 path_q1_3,
                 path_q2,
                 path_q3_1,
                 path_q3_2,
                 path_q3_3,
                 path_q4_ensemble1,
                 path_q4_ensemble3,
                 path_q4_ensemble5,
                 path_q4_horizon5,
                 path_q4_horizon15,
                 path_q4_horizon30,
                 path_q4_numseq100,
                 path_q4_numseq1000,
                 path_q5_random,
                 path_q5_cem_2,
                 path_q5_cem_4
                 ]

    # collect_events(path_list, target_path=data_path)
