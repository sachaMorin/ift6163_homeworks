from collections import OrderedDict

from .base_agent import BaseAgent
from ift6163.models.ff_model import FFModel
from ift6163.policies.MLP_policy import MLPPolicyPG
from ift6163.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from ift6163.infrastructure.replay_buffer import ReplayBuffer
from ift6163.infrastructure.utils import *


class MBAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(MBAgent, self).__init__()

        self.env = env.unwrapped
        self.agent_params = agent_params
        self.ensemble_size = self.agent_params['ensemble_size']

        self.dyn_models = []
        for i in range(self.ensemble_size):
            model = FFModel(
                self.agent_params['ac_dim'],
                self.agent_params['ob_dim'],
                self.agent_params['n_layers'],
                self.agent_params['size'],
                self.agent_params['learning_rate'],
            )
            self.dyn_models.append(model)

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):

        # training a MB agent refers to updating the predictive model using observed state transitions
        # NOTE: each model in the ensemble is trained on a different random batch of size batch_size
        losses = []
        num_data = ob_no.shape[0]
        num_data_per_ens = int(num_data / self.ensemble_size)

        for i in range(self.ensemble_size):

            # select which datapoints to use for this model of the ensemble
            # you might find the num_data_per_env variable defined above useful

            observations = ob_no[i*num_data_per_ens:(i+1)*num_data_per_ens] # TODO(Done)(Q1)
            actions = ac_na[i*num_data_per_ens:(i+1)*num_data_per_ens] # TODO (Done)(Q1)
            next_observations = next_ob_no[i*num_data_per_ens:(i+1)*num_data_per_ens] # TODO (Done)(Q1)


            # Copy this from previous homework
            model = self.dyn_models[i]  # TODO(Done)(Q1)
            log = model.update(observations, actions, next_observations,
                                self.data_statistics)
            loss = log['Training Loss']
            losses.append(loss)
            
        # TODO (Done) Pick a model at random
        model_idx = np.random.choice(len(self.dyn_models))
        model = self.dyn_models[model_idx]

        # TODO (Done) Use that model to generate one additional next_ob_no for every state in ob_no (using the policy distribution)
        # Hint: You may need the env to label the rewards
        # Hint: Keep things on policy
        new_actions = self.actor.get_action(ob_no)
        new_obs = model.get_prediction(ob_no, ac_na, self.data_statistics)
        new_rewards, new_terminals = self.env.get_reward(ob_no, new_actions)
        
        # TODO (Done) add this generated data to the real data
        ob_no_aug = np.concatenate((ob_no, ob_no))
        ac_na_aug = np.concatenate((ac_na, new_actions))
        next_ob_no_aug = np.concatenate((next_ob_no, new_obs))
        re_n_aug = np.concatenate((re_n, new_rewards))
        terminal_n_aug = np.concatenate((terminal_n, new_terminals))

        # TODO (Done) Perform a policy gradient update
        # Update the critic
        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss = self.critic.update(ob_no_aug, ac_na_aug, next_ob_no_aug, re_n_aug, terminal_n_aug)

        advantage = self.estimate_advantage(ob_no_aug, next_ob_no_aug, re_n_aug, terminal_n_aug)

        # Update the actor
        for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
            actor_loss = self.actor.update(ob_no_aug, ac_na_aug, advantage)

        # Hint: Should the critic be trained with this generated data? Try with and without and include your findings in the report.

        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss['Training Loss']
        loss['FD_Loss'] = np.mean(losses)
        return loss

    def add_to_replay_buffer(self, paths, add_sl_noise=False):

        # add data to replay buffer
        self.replay_buffer.add_rollouts(paths, noised=add_sl_noise)

        # get updated mean/std of the data in our replay buffer
        self.data_statistics = {
            'obs_mean': np.mean(self.replay_buffer.obs, axis=0),
            'obs_std': np.std(self.replay_buffer.obs, axis=0),
            'acs_mean': np.mean(self.replay_buffer.acs, axis=0),
            'acs_std': np.std(self.replay_buffer.acs, axis=0),
            'delta_mean': np.mean(
                self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0),
            'delta_std': np.std(
                self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0),
        }

        # update the actor's data_statistics too, so actor.get_action can be calculated correctly
        self.actor.data_statistics = self.data_statistics

    def sample(self, batch_size):
        # NOTE: sampling batch_size * ensemble_size,
        # so each model in our ensemble can get trained on batch_size data
        return self.replay_buffer.sample_random_data(
            batch_size * self.ensemble_size)

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        # TODO (Done) Implement the following pseudocode:
        # 1) query the critic with ob_no, to get V(s)
        # 2) query the critic with next_ob_no, to get V(s')
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)
        terminal_n = np.array(terminal_n)

        v = self.critic.forward_np(ob_no)
        V_prime = self.critic.forward_np(next_ob_no) * (1 - terminal_n)

        q = re_n + self.critic.gamma * V_prime

        adv_n = q - v

        if self.agent_params.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return adv_n

