import os
import sys

# Ensure envs module can be found
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from envs.uav_mimo_env import UavMimoEnv
import numpy as np

def test_environment_loop():
    print("Initializing UAV MIMO Environment...")
    env = UavMimoEnv(num_bs=4, num_uav=10, num_antennas=16)
    obs, info = env.reset(seed=42)
    
    print(f"Observation shape: {obs.shape}")
    print(f"Sample Observation (flattened, first 10 vals): \n{obs[:10]}")
    
    # 随机动作
    print("\nTaking a random step (allocating power)...")
    action = env.action_space.sample()
    
    obs_next, reward, terminated, truncated, info = env.step(action)
    
    print(f"Reward: {reward}")
    print(f"Sum-Rate: {info['sum_rate']}")
    print(f"Individual capacities: \n{info['capacities']}")
    print(f"QoS Violations sum: {info['violations_sum']}")
    print(f"Terminated: {terminated}")

if __name__ == "__main__":
    test_environment_loop()
    print("\nPhase 1 verification passed!")
