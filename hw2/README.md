## Results
Please refer to assignment for all experiment commands.

## Setup

You can run this code on your own machine or on Google Colab (Colab is not completely supported). 

1. **Local option:** If you choose to run locally, you will need to install MuJoCo and some Python packages; see [installation.md](../hw1/installation.md) from homework 1 for instructions. There are two new package requirements (`opencv-python` and `gym[atari]`) beyond what was used in the previous assignments; make sure to install these with `pip install -r requirements.txt` if you are running the assignment locally.

2. **Colab:** The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badges below:

## Complete the code

The following files have blanks to be filled with your solutions from homework 1. The relevant sections are marked with `TODO: get this from Piazza'.

- [infrastructure/rl_trainer.py](ift6163/infrastructure/rl_trainer.py)
- [infrastructure/utils.py](ift6163/infrastructure/utils.py)

You will then need to implement code in the following files:
- [agents/mb_agent.py](ift6163/agents/mb_agent.py)
- [models/ff_model.py](ift6163/models/ff_model.py)
- [policies/MPC_policy.py](ift6163/policies/MPC_policy.py)

The relevant sections are marked with `TODO`.

You may also want to look through [run_hw2_mb.py](run_hw4_mb.py) (if running locally) or [scripts/run_hw2_mb.ipynb](ift6163/scripts/run_hw2_mb.ipynb) (if running on Colab), though you will not need to edit this files beyond changing runtime arguments in the Colab notebook.

See the [assignment PDF](ift6163_hw2.pdf) for more details on what files to edit.

