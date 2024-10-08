def pretrain_neural_network_model_with_val(model, X_train, y_train, X_val, y_val, seed, num_epochs=100, batch_size=32, learning_rate=1e-3, patience=10, scheduler_factor=0.1, scheduler_patience=5):
    # Ensure reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=learning_rate)
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True)

    # Prepare DataLoader for training and validation sets
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    X_val = torch.tensor(X_val.values, dtype=torch.float32)
    y_val = torch.tensor(y_val.values, dtype=torch.float32)
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Initialize variables for early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    model.train()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.flatten(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                val_outputs = model(batch_X_val)
                val_loss += criterion(val_outputs.flatten(), batch_y_val).item()
        
        val_loss /= len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save the best model
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}. Restoring best model from epoch {epoch+1-patience} with val loss of {best_val_loss}")
            model.load_state_dict(best_model_state)  # Restore the best model
            break
    
    return model

def pretrain_neural_network_model_with_sched_reset(model, X_train, y_train, X_val, y_val, seed, num_epochs=100, batch_size=32, learning_rate=1e-3, patience=10, scheduler_factor=0.1, scheduler_patience=5):
    # Ensure reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=learning_rate)
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks

    # Learning rate scheduler (reduce LR when val loss plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True)

    # Prepare DataLoader for training and validation sets
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    X_val = torch.tensor(X_val.values, dtype=torch.float32)
    y_val = torch.tensor(y_val.values, dtype=torch.float32)
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize variables for early stopping
    best_val_loss = float('inf')
    previous_val_loss = float('inf')  # Track previous validation loss
    epochs_without_improvement = 0
    best_model_state = None

    model.train()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.flatten(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                val_outputs = model(batch_X_val)
                val_loss += criterion(val_outputs.flatten(), batch_y_val).item()
        
        val_loss /= len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        # Early stopping logic (now comparing to previous epoch)
        if val_loss < previous_val_loss:  # Improvement compared to previous epoch
            best_val_loss = val_loss if val_loss < best_val_loss else best_val_loss  # Save the best model only when it's the best overall
            best_model_state = model.state_dict()  # Save the best model state
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        previous_val_loss = val_loss  # Update the previous validation loss for next comparison

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}. Restoring best model from epoch {epoch+1-patience} with val loss of {best_val_loss}")
            model.load_state_dict(best_model_state)  # Restore the best model
            break
    
    return model

def pretrain_neural_network_model(model, X, y, seed, num_epochs=10, batch_size=32, learning_rate=1e-3):
    # Ensure reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=learning_rate)
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks    
    # Create DataLoader for batching
    X = torch.tensor(X.values, dtype=torch.float32)  # Ensure the data type is float32 for features
    y = torch.tensor(y.values, dtype=torch.float32)   
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    model.train()  # Set model to training mode


    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()  # Zero the gradients
            
            # Forward pass
            outputs = model(batch_X)
            # Compute loss
            loss = criterion(outputs.flatten(), batch_y)
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            epoch_loss += loss.item()
        
        # Print average loss for the epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}')
    
    return model

def pretrain_neural_network_model_val(model, X, y, seed,num_epochs=50, batch_size=32, learning_rate=1e-3,  early_stopping_patience=5):
    # Prepare for cross-validation
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    spearman_scores = []
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{kf.get_n_splits()}")

        # Split the data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create DataLoader
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize the model, loss function, and optimizer
        # input_dim = X.shape[1]
        # output_dim = 1
        #model = SimpleNN(input_dim, hidden_dim, output_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            # Training phase
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)

            # Validation phase
            model.eval()
            val_predictions = []
            val_targets = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    val_predictions.append(outputs.squeeze().cpu().numpy())
                    val_targets.append(targets.cpu().numpy())
            
            val_predictions = np.concatenate(val_predictions)
            val_targets = np.concatenate(val_targets)
            mse = mean_squared_error(val_targets, val_predictions)
            spearman_corr, _ = spearmanr(val_targets, val_predictions)
            
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, MSE: {mse:.4f}, Spearman Correlation: {spearman_corr:.4f}")
            
            # Early stopping check
            if mse < best_val_loss:
                best_val_loss = mse
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered.")
                break
        
        spearman_scores.append(spearman_corr)
        cv_scores.append(mse)
    print(str(spearman_scores))
    print(str(mse))
    model.eval()
    return model

# class LargeFeatureExtractor(torch.nn.Sequential):
    
#     def __init__(self, data_dim,seed):
#         self.seed = seed
#         random.seed(self.seed)
#         np.random.seed(self.seed)
#         torch.manual_seed(self.seed)
#         super(LargeFeatureExtractor, self).__init__()
#         self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
#         self.add_module('relu1', torch.nn.ReLU())
#         self.add_module('linear2', torch.nn.Linear(1000, 500))
#         self.add_module('relu2', torch.nn.ReLU())
#         self.add_module('linear3', torch.nn.Linear(500, 50))
#         self.add_module('relu3', torch.nn.ReLU())
#         self.add_module('linear4', torch.nn.Linear(50, 2))
#         self.add_module('relu3', torch.nn.ReLU())
#         self.add_module('linear5', torch.nn.Linear(50, 1))
#     def extract(self, x):
#         x = self.linear1(x)  # Pass through the first linear layer
#         x = self.relu1(x)    # Apply the ReLU activation
#         x = self.linear2(x)  # Pass through the second linear layer
#         x = self.relu2(x)    # Apply the ReLU activation
#         x = self.linear3(x)  # Pass through the third linear layer
#         x = self.relu3(x)    # Apply the ReLU activation
#         x = self.linear4(x)  # Pass through the final linear layer (50 -> 2)
#         return x  # Return the final output after processing
    
        
        

# class NeuralNetworkHeart(nn.Module):
#     def __init__(self, input_dim, output_dim = 1):
#         self.input_dim = input_dim
#         super(NeuralNetworkHeart, self).__init__()
#         self.layer1 = nn.Linear(input_dim, 50)
#         self.layer2 = nn.Linear(50, 50)
#         self.layer3 = nn.Linear(50, 50)
#         self.layer4 = nn.Linear(50, output_dim)
#         self.tanh = nn.Tanh()
    
#     def forward(self, x):
#         x = self.tanh(self.layer1(x))
#         x = self.tanh(self.layer2(x))
#         x = self.tanh(self.layer3(x))
#         x = self.layer4(x)
#         return x
    
#     def feature_extractor(self, x):
#         x = self.tanh(self.layer1(x))
#         x = self.tanh(self.layer2(x))
#         x = self.tanh(self.layer3(x))
#         return x