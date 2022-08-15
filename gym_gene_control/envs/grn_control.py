import gym
import wandb
import torch
import numpy as np
import jax.numpy as jnp
import seaborn as sns
from matplotlib import pyplot as plt

import src.zoo_functions
from src.models.expert.classfier_cell_state import MiniCellStateClassifier
import jax_simulator
import src.techinical_noise
import torch.nn as nn


class GRNControlEnvThesis(gym.Env):
    metadata = {'render.modes': ['human']}
    MAX_ACTIONS_VALUE = 50

    def __init__(self,
                 wandb_writer=None,
                 run_rl=0,
                 target_cell_type=0,
                 interactions_glob_filepath=None,
                 regulators_glob_filepath=None,
                 expert_checkpoint_glob_filepath=None,
                 noise_amplitude=0.7,
                 rl_num_intermediate_steps=5,
                 num_cells_to_sim=5,
                 tot_genes=18,
                 tot_cell_types=2, **kwargs):
        """
        :param wandb_writer: wandb init
        :param run_rl: if True will run full MDP, context bandits otherwise
        :param target_cell_type: the index of the cell type to steer
        """

        self.target_cell_type = target_cell_type
        self.run_rl = run_rl
        self.noise_amplitude = noise_amplitude
        self.num_cells_to_sim = num_cells_to_sim
        self.intermediate_steps = rl_num_intermediate_steps

        print('|------- RL' if run_rl else '--- Bandits', f"control cell of index **{target_cell_type}** ------|")

        dataset_dict = {
            "interactions": interactions_glob_filepath,
            "regulators": regulators_glob_filepath,
            "params_outliers_genes_noise": [0.011175966309981848, 2.328873447557661, 0.5011137928428419],
            "params_library_size_noise": [9.961818165607404, 1.2905366314510822],
            "params_dropout_noise": [6.3136458044016655, 62.50611701257209],
            "tot_genes": tot_genes,
            "tot_cell_types": tot_cell_types
        }
        dataset = src.zoo_functions.dataset_namedtuple(*dataset_dict.values())
        classifier = MiniCellStateClassifier(
            num_genes=dataset.tot_genes, num_cell_types=dataset.tot_cell_types).to("cpu")
        loaded_checkpoint = torch.load(expert_checkpoint_glob_filepath, map_location=lambda storage, loc: storage)
        classifier.load_state_dict(loaded_checkpoint)
        classifier.eval()
        self.expert = src.zoo_functions.torch_to_jax(classifier, use_simple_model=True)

        self.sim = jax_simulator.Sim(
            num_genes=dataset.tot_genes, num_cells_types=dataset.tot_cell_types,
            simulation_num_steps=self.num_cells_to_sim,
            interactions_filepath=dataset.interactions, regulators_filepath=dataset.regulators, noise_amplitude=0.7
        )
        self.sim.build()
        self.add_technical_noise_function = None  # src.techinical_noise.AddTechnicalNoiseJax(
        # dataset.tot_genes, dataset.tot_cell_types, params['NUM_SIM_CELLS'],
        # dataset.params_outliers_genes_noise, dataset.params_library_size_noise, dataset.params_dropout_noise
        # )

        action_size = len(self.sim.layers[0])
        self.action_space = gym.spaces.Box(
            low=np.zeros(action_size) + 1e-6,
            high=np.ones(action_size) * self.MAX_ACTIONS_VALUE
        )
        self.observation_space = gym.spaces.Box(
            low=np.zeros((self.sim.num_genes, self.sim.num_cell_types)),
            high=np.ones((self.sim.num_genes, self.sim.num_cell_types)) * np.inf,
        )
        self.initial_state = None
        self.count_resets = 0
        self.count_steps = 0

        self.plotter = wandb_writer

    def step(self, actions, wandb_verbose=True):
        # @jax.jit
        def reward_function(_actions):
            gene_expression = self.sim.run_one_rollout(
                _actions,
                target_idx=self.target_cell_type,
                context_bandits=not self.run_rl
            )
            gene_expression = self.dict_to_array(gene_expression)
            x_T = gene_expression[-1, :, :]

            if self.add_technical_noise_function is not None:
                gene_expression = self.add_technical_noise_function.get_noisy_technical_concentration(
                    gene_expression.T).T
            else:
                gene_expression = x_T

            expert_logits = self.expert(gene_expression.T)
            reward = jnp.mean(expert_logits[:, self.target_cell_type]) - jnp.mean(
                expert_logits[:, 1 - self.target_cell_type], axis=0)

            return reward, gene_expression, expert_logits

        if self.run_rl:
            if self.intermediate_steps > 0:
                reward, x_T, expert_logits = reward_function(actions)
                done = False
                extra_info = {}
                self.intermediate_steps -= 1
            else:
                reward, x_T, expert_logits = reward_function(actions)
                done = True
                extra_info = {}
                self.intermediate_steps = 3

            print(f"step#{self.count_steps}|intermediate#{self.intermediate_steps}|actions:{actions}")

        else:
            reward, x_T, expert_logits = reward_function(actions)
            done = True
            extra_info = {}
            print(f"step#{self.count_steps}|actions:{actions}")

        if self.plotter is not None:
            if reward.dtype == jnp.float32:
                self.plotter.log({"reward/training reward": reward}, step=self.count_steps)
            else:
                self.plotter.log({"reward/training reward": reward[0]}, step=self.count_steps)

            self.plotter.log({"logits/argmax(axis=1).mean()": expert_logits.argmax(axis=1).mean()},
                             step=self.count_steps)
            self.plotter.log({"logits/target_class.mean()": expert_logits[:, self.target_cell_type].mean()},
                             step=self.count_steps)
            self.plotter.log({"logits/not_target_class.mean()": expert_logits[:, 1 - self.target_cell_type].mean()},
                             step=self.count_steps)

            for idx, _action in enumerate(actions):
                self.plotter.log({f"actions/{idx}": actions[idx]}, step=self.count_steps)

            if self.count_steps % 10 == 0:
                heatmap_step = self.count_steps // 10
                heatmap_kwargs = {'linewidth': 0.1, 'cbar_kws': {"shrink": .3}, 'square': True, 'cmap': 'viridis'}

                norm_actions = (actions - actions.mean()) / (actions.std()+0.001)
                heatmap_actions = sns.heatmap(norm_actions.reshape(1, *norm_actions.shape),
                                              **heatmap_kwargs,
                                              yticklabels=['D0' if self.target_cell_type == 0 else 'iPSC'])
                self.plotter.log({"heatmaps/heatmap_actions": wandb.Image(heatmap_actions)}, step=heatmap_step)
                plt.close()

                heatmap_gene_expr = sns.heatmap(x_T.T, yticklabels=['D0', 'iPSC'], **heatmap_kwargs)
                self.plotter.log({"heatmaps/gene_expression": wandb.Image(heatmap_gene_expr)}, step=heatmap_step)
                plt.close()

        self.count_steps += 1
        print(f"training reward:", reward)
        return x_T, reward, done, extra_info

    def reset(self):
        self.count_resets += 1
        if self.initial_state is None:
            actions = np.random.random(self.action_space.shape)
            trajectory = self.dict_to_array(self.sim.run_one_rollout(actions, target_idx=self.target_cell_type))
            last_time_step_of_sim_trajectory = trajectory[-1, :, :]
            self.initial_state = last_time_step_of_sim_trajectory
        return self.initial_state

    def dict_to_array(self, x):
        expr_clean = jnp.stack(tuple([x[gene] for gene in range(self.sim.num_genes)])).swapaxes(0, 1)
        return expr_clean

    def render(self, mode='human'):
        """TODO: this should visualize the graph."""
        raise NotImplementedError

    def close(self):
        raise NotImplemented


class GRNControlEnvDS4(gym.Env):
    target_gene_type = 0
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.noise_amplitude = 0.8
        self.num_cells_to_sim = 10
        self.MAX_ACTIONS_VALUE = 10

        params = {'num_genes': 100, 'NUM_SIM_CELLS': 100}
        dataset_dict = src.zoo_functions.open_datasets_json(return_specific_key='DS4')
        dataset = src.zoo_functions.dataset_namedtuple(*dataset_dict.values())

        expert_checkpoint_filepath = "src/models/expert/checkpoints/classifier_ds4.pth"
        classifier = src.models.expert.classfier_cell_state.CellStateClassifier(num_genes=dataset.tot_genes,
                                                                                num_cell_types=dataset.tot_cell_types).to(
            "cpu")
        loaded_checkpoint = torch.load(expert_checkpoint_filepath, map_location=lambda storage, loc: storage)
        classifier.load_state_dict(loaded_checkpoint)
        classifier.eval()
        self.expert = src.zoo_functions.torch_to_jax(classifier)

        self.sim = jax_simulator.Sim(
            num_genes=dataset.tot_genes, num_cells_types=dataset.tot_cell_types,
            simulation_num_steps=params['NUM_SIM_CELLS'],
            interactions_filepath=dataset.interactions, regulators_filepath=dataset.regulators, noise_amplitude=0.7
        )
        self.sim.build()
        self.add_technical_noise_function = src.techinical_noise.AddTechnicalNoiseJax(
            dataset.tot_genes, dataset.tot_cell_types, params['NUM_SIM_CELLS'],
            dataset.params_outliers_genes_noise, dataset.params_library_size_noise, dataset.params_dropout_noise
        )

        action_size = len(self.sim.layers[0])
        self.action_space = gym.spaces.Box(
            low=np.zeros(action_size) + 1e-6,
            high=np.ones(action_size) * self.MAX_ACTIONS_VALUE
        )
        self.observation_space = gym.spaces.Box(
            low=np.zeros((self.sim.num_genes, self.sim.num_cell_types)),
            high=np.ones((self.sim.num_genes, self.sim.num_cell_types)) * np.inf,
        )
        self.initial_state = None

    def step(self, action):
        # @jax.jit
        def loss_fn(actions):
            gene_expression = self.sim.run_one_rollout(actions)
            gene_expression = self.dict_to_array(gene_expression)
            x_T = gene_expression[-1, :, :]

            if self.add_technical_noise_function is not None:
                gene_expression = self.add_technical_noise_function.get_noisy_technical_concentration(
                    gene_expression.T).T
            else:
                gene_expression = jnp.concatenate(gene_expression, axis=1).T

            expert_log_probs = self.expert(gene_expression)
            cross_entropy = jnp.mean(expert_log_probs[:, self.target_gene_type])
            return -cross_entropy, x_T

        reward, x_T = loss_fn(action)
        done = True
        extra_info = {}

        return x_T, reward, done, extra_info

    def reset(self):
        if self.initial_state is None:
            action = np.random.random((self.action_space.shape)) * 10
            trajectory = self.dict_to_array(self.sim.run_one_rollout(action))
            self.initial_state = trajectory[-1, :, :]  # last_time_step of the simulation trajectory
        return self.initial_state

    def dict_to_array(self, x):
        expr_clean = jnp.stack(tuple([x[gene] for gene in range(self.sim.num_genes)])).swapaxes(0, 1)
        return expr_clean

    def render(self, mode='human'):
        raise NotImplemented

    def close(self):
        raise NotImplemented


if __name__ == '__main__':
    env = GRNControlEnvDS4()
    x = env.reset()
    print(x)
    for _ in range(100):
        action = env.action_space.sample()
        x, reward, done, extra_info = env.step(action)
        print("reward...    ", reward)