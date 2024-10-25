
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import re
from torch.utils.data import DataLoader, TensorDataset

class PredictionModelEPI(nn.Module):
    def __init__(self, input_size,  output_size, hidden_size=32, embedding_size=32, num_trajectory_steps=10):
        super(PredictionModelEPI, self).__init__()
        self.num_trajectory_steps = num_trajectory_steps
        self.input_size = input_size
        action_size = input_size - output_size
        obs_size = output_size
        self.trajectory_total_size = obs_size*(self.num_trajectory_steps+ 1) + self.num_trajectory_steps * (action_size)
        
        # Embedding part
        self.embedding = nn.Sequential(
            nn.Linear(self.trajectory_total_size, embedding_size),  # Embedding layer
            nn.ReLU()
        )
        
        # Fully connected layers after embedding
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_size+input_size, hidden_size),  # First fully connected layer
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  # Second fully connected layer
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        embedding_vector = x[:, :self.trajectory_total_size]
    
        # Pass through embedding model
        embedding_output = self.embedding(embedding_vector)
        
        # Concatenate the output of embedding and the input
        x = torch.cat((embedding_output, x[:, self.trajectory_total_size:]), dim=-1)  # Concatenate along the last dimension
        
        # Pass through fully connected layers
        x = self.fc_layers(x)
        return x

    def get_embedding(self, x):
        # Only extract the embedding part (until the embedding model)
        return self.embedding(x)


class VanillaPredictionModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(VanillaPredictionModel, self).__init__()
        # Fully connected layers for prediction (4 layers with 128 neurons each)
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # First fully connected layer
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  # Second fully connected layer
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  # third fully connected layer
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  # 4th fully connected layer
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # Pass through the fully connected layers
        return self.fc_layers(x)


def extract_data(data_split_dict):

    train_data = pd.DataFrame()
    val_data = pd.DataFrame() 
    for env, data in data_split_dict.items():
        train_data_env = data['train']
        val_data_env = data['val']
        train_data = pd.concat([train_data, train_data_env], ignore_index=True)
        val_data = pd.concat([val_data, val_data_env], ignore_index=True)
    return train_data, val_data
        


    

def dataloading(data_dir):

    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    # Dictionary to hold the train and validation sets
    data_splits = {}

    for file in csv_files:
        # Load the CSV file
        df = pd.read_csv(file, sep=';')
        
        # Split into training and validation sets
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Store the splits in the dictionary using the filename as the key
        data_splits[os.path.basename(file)] = {'train': train_df, 'val': val_df}
        # Get training columns
    train_pattern = r'^(state_|action_)'
    target_pattern = r'^(next_)'

    train_columns = [col for col in train_df.columns if re.match(train_pattern, col)]
    # Get target columns
    target_columns = [col for col in train_df.columns if re.match(target_pattern, col)]
    input_size = len(train_columns)
    output_size = len(target_columns)
    return data_splits, input_size, output_size


def train_model(model, train_df, batch_size=64, epochs=100, trajectories_dict=None):
    criterion = nn.MSELoss()  # Using Mean Squared Error
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_pattern = r'^(state_|action_)'
    target_pattern = r'^(next_)'

    # Get training columns
    train_columns = [col for col in train_df.columns if re.match(train_pattern, col)]
    
    # Get target columns
    target_columns = [col for col in train_df.columns if re.match(target_pattern, col)]

    # Convert the data to tensors
    inputs = torch.tensor(train_df[train_columns].values, dtype=torch.float32)
    targets = torch.tensor(train_df[target_columns].values, dtype=torch.float32)

    if trajectories_dict is not None:
        trajectories_df = pd.DataFrame(list(trajectories_dict.items()), columns=['env_context', 'trajectory'])
        # Merge the original DataFrame with the trajectories DataFrame
        merged_df = train_df.merge(trajectories_df, on='env_context', how='left')
        trajectories = merged_df['trajectory']
        expanded_trajectories = pd.DataFrame(trajectories.tolist(), index=merged_df.index)
        # Rename the new trajectory columns (optional)
        expanded_trajectories.columns = [f'trajectory_{i}' for i in range(expanded_trajectories.shape[1])]
        trajectories = torch.tensor(expanded_trajectories.values, dtype=torch.float32)
        inputs = torch.cat((trajectories, inputs), dim=-1)

    # Create a TensorDataset and DataLoader for batch processing
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0  # To accumulate loss over the epoch
        for batch_inputs, batch_targets in dataloader:
            # Forward pass
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:  # Print every 10 epochs
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

    return model  # Return the final loss after training


def evaluate_model(model, val_df, env_identifier, batch_size=64, embedding=None):
    train_pattern = r'^(state_|action_)'
    target_pattern = r'^(next_)'

    # Get training columns
    train_columns = [col for col in val_df.columns if re.match(train_pattern, col)]
    # Get target columns
    target_columns = [col for col in val_df.columns if re.match(target_pattern, col)]

    # Filter validation data by environment
    val_env_df = val_df[val_df['env_context'] == str(env_identifier)]

    # Convert to tensors
    inputs = torch.tensor(val_env_df[train_columns].values, dtype=torch.float32)
    targets = torch.tensor(val_env_df[target_columns].values, dtype=torch.float32)
    if embedding is not None:
        inputs = torch.cat((embedding.repeat(inputs.size(0), 1), inputs), dim=1)

    model.eval()
    mse_criterion = nn.MSELoss(reduction='mean')
    total_loss = 0.0
    num_batches = 0

    # Evaluate in batches
    for i in range(0, len(inputs), batch_size):
        input_batch = inputs[i:i + batch_size]
        target_batch = targets[i:i + batch_size]

        with torch.no_grad():
            predictions = model(input_batch)
            loss = mse_criterion(predictions, target_batch)
            total_loss += loss.item()
            num_batches += 1

    # Return the average loss, should it be len batches ?
    avg_loss = total_loss / num_batches
    #print(f'Evaluation MSE for environment {env_identifier}: {avg_loss:.4f}')
    return avg_loss







