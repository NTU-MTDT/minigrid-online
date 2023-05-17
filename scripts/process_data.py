import utils
import json
import pickle

from tqdm.auto import tqdm
import numpy as np

datasets = [
    [
        "dataset_SingleTaskGoal-v0_0_1000",
        # "dataset_SingleTaskGoal-v0_30_1000"
    ],
    ["dataset_SingleTaskRight-v0_5_1000"],
    ["dataset_SingleTaskLeft-v0_5_1000"],
]
output_file = "threetasks"

env_name = "MultiTask-v0"  # Should be multi task

# retrieve observation and reward
utils.seed(0)
env = utils.make_env(env_name, render_mode="rgb_array")
action_dim = env.action_space.n
reward_dim = env.reward_dimension

assert reward_dim == len(datasets)


def collect_experience(env, task_id, trajectory):
    seed = trajectory["seed"]
    utils.seed(seed)
    obs, _ = env.reset(seed=seed)
    experience = {
        "observations": [obs["image"]],
        "actions": [],
        "rewards": [],
        "dones": [],
    }
    for action in trajectory["actions"]:
        obs, reward, terminated, truncated, _ = env.step(action)
        # print(action, reward, terminated, truncated)
        done = terminated | truncated

        experience["observations"].append(obs["image"])
        experience["actions"].append(np.eye(action_dim)[action])  # one-hot
        experience["rewards"].append(reward)
        experience["dones"].append(done)
        if done:
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
        with open(f"dataset/{dataset}.json", "r") as f:
            data = json.load(f)

        for trajectory in tqdm(data["trajectories"], desc=dataset):
            experience = collect_experience(
                env=env, task_id=task_id, trajectory=trajectory
            )
            experiences.append(experience)
            total_timesteps += experience["observations"].shape[0]

print(f"Total experiences: {len(experiences)}")
print(f"Total timesteps: {total_timesteps}")
print(f"Observation shape: {experiences[0]['observations'][0].shape}")

# save data
import pickle

with open(f"dataset/{output_file}.pkl", "wb") as f:
    pickle.dump(experiences, f)
