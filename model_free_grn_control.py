import getpass

import experiment_buddy

import gym_gene_control  # noqa
from src.rl_agents import train_a2c  # noqa
from src.rl_agents import train_ddpg  # noqa
from src.rl_agents import train_ppo  # noqa
from src.rl_agents import train_sac  # noqa
from src.rl_agents import train_td3  # noqa
from src.zoo_functions import open_datasets_json


def main():
    whoami = getpass.getuser()
    if whoami == 'ionelia':
        filepath = 'data/rl_agents_params.json'
    elif whoami == 'ionelia.buzatu':
        filepath = "/network/projects/_groups/grn_control/graphD0D18genes#18/thesis_rl_agents_params.json"

    print(f"Loading JSON file from: {filepath}")
    json_params = open_datasets_json(filepath=filepath)
    generals_params = json_params['generals']

    # agent_to_train = [train_a2c, train_td3, train_sac, train_ppo, train_ddpg]
    # agent_to_train = [train_ppo, train_ddpg]
    agent_to_train = [train_ppo]

    config = {
        "policy_type": "MlpPolicy",
        "cell_id_to_steer": generals_params['target_cell_type'],
        "runRL(bool)": generals_params['run_rl']
    }

    experiment_buddy.register_defaults(config)

    for agent in agent_to_train:

        agent_params_kwargs = json_params[agent.__name__]
        run_suffix = f"#steer:{generals_params['target_cell_type']}#runRL:{generals_params['run_rl']}"
        wandb_run_name = f"#{run_suffix}"

        buddy = experiment_buddy.deploy(
            host=generals_params['host'],
            disabled=False,
            wandb_kwargs={
                'sync_tensorboard': True,
                'monitor_gym': True,
                'save_code': True,
                'entity': 'control-grn',
                'project': 'RL',
                'reinit': True
            },
            wandb_run_name=f"RL#2M#steer{generals_params['target_cell_type']}#{str(agent.__name__)}",
            extra_modules=["cuda/11.1/nccl/2.10", "cudatoolkit/11.1", "cuda/11.1/cudnn/8.1"]
            
        )
        run = buddy.run
        print(f"running ***{wandb_run_name}*** in progress...")

        agent(run, agent_params_kwargs, json_params['generals'])

        print("Done training", agent.__name__)


if __name__ == '__main__':
    main()