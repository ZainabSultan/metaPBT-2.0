
from DKL.EPI.models import PredictionModelEPI, VanillaPredictionModel, dataloading, train_model, extract_data
from DKL.EPI.epi_envs import CARLMountainCarWrapper, generate_trajs, CARLLunarLanderWrapper
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
import numpy as np
from DKL.EPI.CARLWrapper import env_creator
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.cm as cm


env_name='CARLLunarLander'
feature = 'LEG_H'
data_dir = '/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/DKL/EPI/datasets/CARLMountainCar'
data_dir = '/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/DKL/EPI/datasets/CARLLunarLander'
data, in_dim, out_dim = dataloading(data_dir)
defaults_cartpole = {'length': 0.5, 'tau': 0.02, 'gravity': 9.8}
defaults_mcd = {'gravity': 0.0025}
defaults_lunar_lander = {
    "FPS": 50,
    "SCALE": 30.0,
    "MAIN_ENGINE_POWER": 13.0,
    "SIDE_ENGINE_POWER": 0.6,
    "INITIAL_RANDOM": 1000.0,
    "GRAVITY_X": 0,
    "GRAVITY_Y": -10,
    "LEG_AWAY": 20,
    "LEG_DOWN": 18,
    "LEG_W": 2,
    "LEG_H": 8,
    "LEG_SPRING_TORQUE": 40,
    "SIDE_ENGINE_HEIGHT": 14.0,
    "SIDE_ENGINE_AWAY": 12.0,
    "VIEWPORT_W": 600,
    "VIEWPORT_H": 400
}
defaults_bipedal = {
    "FPS": 50,
    "SCALE": 30.0,
    "GRAVITY_X": 0,
    "GRAVITY_Y": -10,
    "FRICTION": 2.5,
    "TERRAIN_STEP": 14 / 30.0,
    "TERRAIN_LENGTH": 200,
    "TERRAIN_HEIGHT": 600 / 30 / 4,
    "TERRAIN_GRASS": 10,
    "TERRAIN_STARTPAD": 20,
    "MOTORS_TORQUE": 80,
    "SPEED_HIP": 4,
    "SPEED_KNEE": 6,
    "LIDAR_RANGE": 160 / 30.0,
    "LEG_DOWN": -8 / 30.0,
    "LEG_W": 8 / 30.0,
    "LEG_H": 34 / 30.0,
    "INITIAL_RANDOM": 5,
    "VIEWPORT_W": 600,
    "VIEWPORT_H": 400
}
def plot(embeddings, save_path):
    # Use t-SNE for dimensionality reduction to 2D
    # Convert dictionary to lists of labels and embeddings for easier manipulation
    labels = list(embeddings_dict.keys())
    embeddings = np.array(list(embeddings_dict.values()))

    # Dimensionality reduction to 2D
    perplexity = min(5, len(embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Generate unique colors for each label
    num_embeddings = len(labels)
    colors = cm.rainbow(np.linspace(0, 1, num_embeddings))

    # Plot each embedding with its label and a unique color
    plt.figure(figsize=(12, 10))
    for i, label in enumerate(labels):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color=colors[i], label=label, s=50)

    # Add labels, legend, and title
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("t-SNE Visualization of Embeddings with Labels")
    plt.legend(loc='best')
    plt.savefig(f'{save_path}/embeddings_fig.png')
# Determine the default feature value based on the environment and feature
if env_name == 'CARLMountainCar':
    DEFAULT_VALUE = defaults_mcd[feature]
elif env_name == 'CARLCartPole':
    DEFAULT_VALUE = defaults_cartpole[feature]
elif env_name == 'CARLBipedalWalker':
    DEFAULT_VALUE = defaults_bipedal[feature]
elif env_name == 'CARLLunarLander':
    DEFAULT_VALUE = defaults_lunar_lander[feature]

# Set job time and limits based on environment difficulty

# Set the range of feature values
feature_values = np.array([
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
    1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 
    2.1, 2.2
]) * DEFAULT_VALUE
contexts = {}
for i in range(len(feature_values)):
    contexts[i] = {feature : feature_values[i]}

contexts = {0: {'LEG_H': 0.8}, 1: {'LEG_H': 5.6}, 2:{'LEG_H': 10.4}}

epi_embedding_conditioned_model = PredictionModelEPI(input_size = in_dim, output_size=out_dim)
vanilla_model = VanillaPredictionModel(input_size = in_dim, output_size=out_dim)
#contexts = {0 : {'gravity' : 0.00025}, 1:{'gravity': 0.00035}}
n_interleaved_training_epochs = 50

for i in range(n_interleaved_training_epochs):
    train_df, val_df = extract_data(data)
    print(train_df.describe())
    print(val_df.describe())
    env = CARLLunarLanderWrapper(contexts = contexts, epi_embedding_conditioned_model = epi_embedding_conditioned_model,
                                vanilla_model = vanilla_model, val_data=val_df)
    # train on random policy 
    model = PPO(MlpPolicy, env, verbose=2, batch_size=100)
    model.learn(progress_bar=True, total_timesteps=100)
    # wrapped in monitor
    env = CARLLunarLanderWrapper(contexts = contexts, epi_embedding_conditioned_model = epi_embedding_conditioned_model,
                                vanilla_model = vanilla_model, val_data=val_df)
    # generate batch of trajs
    batch = generate_trajs(model, env)
    print(batch.keys())
    # train the models
    #print('epi model')
    epi_embedding_conditioned_model = train_model(epi_embedding_conditioned_model, train_df, trajectories_dict = batch, epochs=100)
    #print('vanilla model')
    vanilla_model = train_model(vanilla_model, train_df, epochs=100)
    

model_path = f'/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/DKL/EPI/trajectory_generating_policies/{env_name}'
model.save(model_path)
model_save_path = f"/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/DKL/EPI/embedding_models/{env_name}.pth"
# Save the model state dictionary
torch.save(epi_embedding_conditioned_model.state_dict(), model_save_path)
# use them to train a policy
# test to see the embeddings
# model.load(model_save_path)
# epi_embedding_conditioned_model = YourModelClass(*args, **kwargs)  # Replace with the actual model class and init parameters
epi_embedding_conditioned_model.eval()
#     # Load the saved weights
# epi_embedding_conditioned_model.load_state_dict(torch.load(model_save_path))
# epi_embedding_conditioned_model.eval() 
embeddings_dict = {}

for _, context in contexts.items():
    for inner_key, inner_value in context.items():
        env_descriptor = f"{inner_key.lower()}: {inner_value}"
    
    env_config = context | {'env_name' :  'CARLLunarLander'}
    env = env_creator(env_config)
    trajectory_dict = generate_trajs(model, env)
    trajectory= trajectory_dict[str(context)]
    trajectory =torch.tensor(trajectory, dtype=torch.float32)
    embedding = epi_embedding_conditioned_model.get_embedding(trajectory)
    if hasattr(embedding, 'detach'):
        embedding = embedding.detach().cpu().numpy()
        print(embedding.shape)
    embeddings_dict[env_descriptor] = embedding

plot(embeddings_dict, model_path)


    
        
    


# Assume embeddings is a 2D array of shape (num_samples, embedding_dim)
# Example embeddings array for demonstration:
# embeddings = np.random.rand(10, 100)  # 10 embeddings of dimension 100



# need to match context from dict with context in df and then split it before training