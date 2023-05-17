import argparse
import numpy as np
import random
import json

import utils
from utils import device

from tqdm import tqdm
import seaborn as sns

print(f"Device: {device}\n")

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True, help="name of the environment (REQUIRED)")
parser.add_argument(
    "--model", required=True, help="name of the trained model (REQUIRED)"
)  # Should be single task
parser.add_argument(
    "--episodes", type=int, default=100, help="number of episodes (default: 100)"
)
parser.add_argument(
    "--noise", type=float, default=0.0, help="noise prob (default: 0.0)"
)


args = parser.parse_args()
env_name = args.env
noise_prob = args.noise
num_episodes = args.episodes
output_file = f"dataset_{env_name}_{int(noise_prob*100)}_{num_episodes}.json"

# Load environment
env = utils.make_env(env_name, render_mode="rgb_array", max_steps=100)
print("Environment loaded\n")

# Load agent
model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(
    env.observation_space,
    env.action_space,
    model_dir,
    argmax=False,
    use_memory=False,
    use_text=False,
)
print("Agent loaded\n")

# Run the agent

trajectories = []

seed = 100000
pbar = tqdm(total=num_episodes)
while len(trajectories) < num_episodes:
    seed += 1
    utils.seed(seed)
    obs, _ = env.reset(seed=seed)

    returnn = 0
    action_seq = []
    while True:
        action = agent.get_action(obs)

        if random.random() < noise_prob:
            action_candidate = [i for i in range(3) if i != action]
            action = random.choice(action_candidate)

        action_seq.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        returnn += reward
        done = terminated | truncated

        if done:
            break

    timestep = len(action_seq)
    if timestep > 3:
        pbar.update(1)
        trajectories.append(
            {
                "seed": seed,
                "timesteps": timestep,
                "actions": [int(a) for a in action_seq],
                "return": returnn,
            }
        )

pbar.close()
trajectories.sort(key=lambda x: x["return"], reverse=True)
returns = [t["return"] for t in trajectories]
timesteps = [t["timesteps"] for t in trajectories]

data = {
    "env_name": env_name,
    "noise_prob": noise_prob,
    "total_episodes": len(trajectories),
    "timesteps": {
        "sum": sum(timesteps),
        "min": min(timesteps),
        "max": max(timesteps),
        "median": np.median(timesteps),
        "mean": np.mean(timesteps),
        "std": np.std(timesteps),
    },
    "return": {
        "min": min(returns),
        "max": max(returns),
        "median": np.median(returns),
        "mean": np.mean(returns),
        "std": np.std(returns),
    },
    "trajectories": sorted(trajectories, key=lambda x: x["return"], reverse=True),
}

with open(f"dataset/{output_file}", "w") as f:
    json.dump(data, f, indent=4)

plt = sns.displot(returns, bins=20, kde=True, binrange=(0, 1))
plt.fig.savefig(f"dataset/returns_{env_name}_{int(noise_prob*100)}_{num_episodes}.png")

plt = sns.displot(timesteps, kde=True, binrange=(0, 512))
plt.fig.savefig(f"dataset/timesteps_{env_name}_{int(noise_prob*100)}_{num_episodes}.png")
