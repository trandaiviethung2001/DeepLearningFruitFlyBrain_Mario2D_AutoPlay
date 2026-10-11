#!/bin/bash
cd "$(dirname "$0")"

clear
echo "======================================================================"
echo "🧠 LIVE ONLINE REINFORCEMENT LEARNING (LEARNING FROM MISTAKES LIVE)"
echo "======================================================================"
echo "Mario is playing and learning LIVE on your screen!"
echo "Mechanism: After every round, PPO updates the synaptic weights based on"
echo "reward gains and death penalties (Dopaminergic TD Error Backpropagation)."
echo "======================================================================"
echo ""

./.venv/bin/python src/ppo_trainer.py --model-type connectome --render --total-timesteps 10000 --rollout-steps 512 --frame-stack 4 --lr 1e-4 --use-dopamine --ent-coef 0.05

echo ""
echo "Live training session complete!"
read -p "Press [Enter] to exit..."
