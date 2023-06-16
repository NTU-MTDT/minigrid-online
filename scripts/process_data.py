import utils
import json
import pickle
import os

from tqdm.auto import tqdm
import numpy as np

datasets = [
    [
        "dataset_SingleTaskGoal-v0_0_1000",
        "dataset_SingleTaskGoal-v0_5_1000",
    ],
    [
        "dataset_SingleTaskLava-v0_0_1000",
        "dataset_SingleTaskLava-v0_5_1000",
    ],
]

output_file = "lava_goal"

# datasets = [
#     [
#         "dataset_SingleTaskGoal-v0_0_1000",
#         # "dataset_SingleTaskGoal-v0_5_1000"
#     ],
# ]

# output_file = "single_expert"

# env_name = "SingleTask-v0"  # Should be multi task

# retrieve observation and reward
utils.seed(0)

# assert reward_dim == len(datasets)

if not os.path.exists(f"dataset/{output_file}"):
    os.system(f"mkdir -p dataset/{output_file}")


def collect_experience(env, task_id, trajectory):
    action_dim = env.action_space.n
    reward_dim = env.reward_dimension

    seed = trajectory["seed"]
    utils.seed(seed)
    obs, _ = env.reset(seed=seed)
    experience = {
        "observations": [obs["image"]],
        "actions": [],
        "mask": [],
        "rewards": [],
        "dones": [],
    }
    for action in trajectory["actions"]:
        obs, reward, terminated, truncated, _ = env.step(action)
        # print(action, reward, terminated, truncated)
        done = terminated | truncated

        # print(action, reward, terminated, truncated, done, env.agent_pos)
        experience["seed"] = seed
        experience["observations"].append(obs["image"])
        experience["actions"].append(np.eye(action_dim)[action])  # one-hot
        experience["mask"].append(np.eye(len(datasets))[task_id])  # one-hot
        experience["rewards"].append(reward * np.eye(len(datasets))[task_id])
        experience["dones"].append(done)
        if done:
            # print(env.action_history)
            # print(env.agent_pos)
            # print(experience["rewards"][-5:])
            # print(terminated)
            break

    assert len(experience["actions"]) == trajectory["timesteps"]

    # Turn to numpy
    experience["observations"] = experience["observations"][:-1]
    # print("seed", seed, "len", len(experience["observations"]))
    for k in experience.keys():
        experience[k] = np.array(experience[k])
        # print(k, experience[k].shape)

    return experience


experiences = []
total_timesteps = 0

for task_id, task in enumerate(datasets):
    for dataset in task:
        os.system(f"cp dataset/{dataset}.json dataset/{output_file}/{dataset}.json")
        with open(f"dataset/{dataset}.json", "r") as f:
            data = json.load(f)

        env_name = data["env_name"]
        env = utils.make_env(env_name, render_mode="rgb_array")
        for trajectory in tqdm(data["trajectories"], desc=dataset):
            experience = collect_experience(
                env=env, task_id=task_id, trajectory=trajectory
            )
            # print(np.sum(np.array(experience["rewards"]), axis=0))
            # print(experience)
            # exit()
            experiences.append(experience)
            total_timesteps += experience["observations"].shape[0]

print(f"Total experiences: {len(experiences)}")
print(f"Total timesteps: {total_timesteps}")
print(f"Observation shape: {experiences[0]['observations'][0].shape}")

# save data
import pickle

with open(f"dataset/{output_file}/{output_file}.pkl", "wb") as f:
    pickle.dump(experiences, f)
