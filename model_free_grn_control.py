import experiment_buddy
import gym_gene_control  # noqa
from src.rl_agents import train_a2c # noqa
from src.rl_agents import train_ddpg # noqa
from src.rl_agents import train_ppo # noqa
from src.rl_agents import train_sac # noqa
from src.rl_agents import train_td3 # noqa
from src.zoo_functions import open_datasets_json
import getpass


def main():
    whoami = getpass.getuser()
    if whoami == 'ionelia':
        filepath='data/rl_agents_params.json'
    elif whoami == 'ionelia.buzatu':
        filepath = "todo/rl_agents_params.json"
    json_params = open_datasets_json(filepath=filepath)

    agent_to_train = [train_a2c, train_td3, train_sac, train_ppo, train_ddpg]

    config = {
        "policy_type": "MlpPolicy",
    }

    experiment_buddy.register_defaults(config)

    for agent in agent_to_train:

        agent_params_kwargs = json_params[agent.__name__]
        generals_params = json_params['generals']
        run_prefix = f"bandits#{agent.__name__}"
        run_suffix = f"#steer:{generals_params['target_cell_type']}#runRL:{generals_params['run_rl']}"
        wandb_run_name = f"{run_prefix}#{run_suffix}"
        print(f"run ***{wandb_run_name}*** in progress...")

        buddy = experiment_buddy.deploy(
            host=generals_params['host'], disabled=False,
            wandb_kwargs={
                'sync_tensorboard': False,
                'monitor_gym': True,
                'save_code': True,
                'entity': 'control-grn',
                'project': 'context-bandits',
                'reinit': True
            },
            wandb_run_name=wandb_run_name
        )
        run = buddy.run

        agent(run, agent_params_kwargs, json_params['generals'])

        print("Done training", agent.__name__)


if __name__ == '__main__':
    main()