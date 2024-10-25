import pandas as pd

ppo_bipedal_csv= 'box2d_bipedal_walker_ppo.csv'

df = pd.read_csv(f"hf://datasets/autorl-org/arlbench/{ppo_bipedal_csv}")
print(df.describe())