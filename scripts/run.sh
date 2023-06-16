env=$1
mode=$2

noise=0.00

export CUDA_VISIBLE_DEVICES=0

if [ $mode == "train" ]; then
    echo "Training $env"
    python3 -m scripts.train --algo ppo --env $env --model $env --procs 20
elif [ $mode == "eval" ]; then
    echo "Evaluating $env"
    python3 -m scripts.evaluate --env $env --model $env --seed 100000 --episodes 1000 --procs 40 --noise $noise
elif [ $mode == "visual" ]; then
    echo "Visualizing for $env"
    python3 -m scripts.visualize --env $env --model $env --gif $env --episodes 10 --seed 100000 --pause 0.05 --noise $noise
elif [ $mode == "gen" ]; then
    echo "Generating data for $env"
    python3 -m scripts.gen_dataset --env $env --model $env --episodes 1000 --noise $noise
else
    echo "Invalid mode"
    exit 1
fi
