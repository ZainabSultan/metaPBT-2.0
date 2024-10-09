import tracemalloc
import torch
import torch.nn as nn

import numpy as np

import torch
import matplotlib.pyplot as plt
import tqdm.notebook
import gpytorch
from ray import train

import torch
import logging
from scipy.optimize import minimize
logger = logging.getLogger(__name__)
import random
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)

def pretrain_neural_network_model(model, X_train, y_train, X_val, y_val, seed, num_epochs=100, batch_size=64, learning_rate=1e-3, patience=10, scheduler_factor=0.1, scheduler_patience=5, use_scheduler=False):
    # Ensure reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=learning_rate)
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks

    # Learning rate scheduler (reduce LR when val loss plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True)
    
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize variables for early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

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
        if use_scheduler:
            scheduler.step(val_loss)
        
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
    predictions = model(X_val).detach().cpu().numpy().flatten()
    residuals = y_val.numpy() - predictions
    # plt.hist(residuals, bins=30)
    # plt.title("Residual Distribution")
    # plt.savefig('Residual Distribution.png')
    
    return model

class DynamicNN(nn.Module):

    '''

    # Example usage:
    # Architecture is a list of layers with their types and other configurations
    architecture = [
        {'type': 'linear', 'out_dim': 1000},
        {'type': 'relu'},
        {'type': 'linear', 'out_dim': 500},
        {'type': 'relu'},
        {'type': 'linear', 'out_dim': 50},
        {'type': 'relu'},
        {'type': 'linear', 'out_dim': 2}  # Feature extraction output size
    ]

    # Dynamically create the neural network
    #input_dim = 10  # Example input dimension
    #model = DynamicNN(architecture, input_dim, seed=42)

    # Print the architecture of the dynamically created model
    #print(model)
    
    
    '''
    def __init__(self, architecture, input_dim, seed, joint_gp_training_phase=False):
        super(DynamicNN, self).__init__()
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        self.joint_gp_training_phase = joint_gp_training_phase
        self.layers = nn.ModuleList()  # Use ModuleList to dynamically append layers
        
        prev_dim = input_dim
        for layer_info in architecture:
            layer_type = layer_info['type']
            
            if layer_type == 'linear':
                out_dim = layer_info['out_dim']
                self.layers.append(nn.Linear(prev_dim, out_dim))
                prev_dim = out_dim
            
            elif layer_type == 'relu':
                self.layers.append(nn.ReLU())
            
            # Add more types as needed (e.g., dropout, batchnorm)
        
        # Define a final output layer if needed
        self.final_layer = nn.Linear(prev_dim, 1)  # Output size 1 for regression or binary classification

    def extract(self, x):
        # Pass input through the layers (up to the last one before the final output)
        for layer in self.layers:
            x = layer(x)
        return x
    
    def forward(self, x):
        x = self.extract(x)
        if not self.joint_gp_training_phase:
            x = self.final_layer(x)  # Apply the final output layer in some training phases
        return x
    

class MultiTaskGPModel_DKL(gpytorch.models.ApproximateGP): # approx vs exact
    def __init__(self, inducing_points, input_dim, num_tasks, feature_extractor):
        super(MultiTaskGPModel_DKL, self).__init__(gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0)))

        self.num_tasks = num_tasks
        self.inducing_points = inducing_points
        self.feature_extractor = feature_extractor

        # Task-specific GPs
        self.gp_layers = nn.ModuleList([
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim) for _ in range(num_tasks)
        ])

    def forward(self, x):
        z = self.feature_extractor(x)  # Extract features
        return [gp_layer(z) for gp_layer in self.gp_layers]
    


class GPRegressionModel_DKL(gpytorch.models.ExactGP):

        def __init__(self, train_x, train_y, likelihood, seed,feature_extractor= None):
            self.seed=seed
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            super(GPRegressionModel_DKL, self).__init__(train_x, train_y, likelihood)


            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.GridInterpolationKernel( # this is to approx kernel
                #gpytorch.kernels.ScaleKernel(gpytorch.kernels.spectral_mixture_kernel(num_mixtures=4)),
                gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=2), # bc SM shouldnt be combined with scale kernels
                # TODO set batch size??
                num_dims=2, grid_size=100
            )
            self.feature_extractor = feature_extractor
            self.X = train_x

            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(0, 1.) # was -1 but changed to match the other parts  

        def forward(self, x):
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            # We're first putting our data through a deep net (feature extractor)
            projected_x = self.feature_extractor(x)
            projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
class GPRegressionModel_DKL(gpytorch.models.ExactGP):

        

        def __init__(self, train_x, train_y, likelihood, seed,feature_extractor, append_sim_column=True):
            
            self.seed=seed
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            super(GPRegressionModel_DKL, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.GridInterpolationKernel( # this is to approx kernel
                #gpytorch.kernels.ScaleKernel(gpytorch.kernels.spectral_mixture_kernel(num_mixtures=4)),
                gpytorch.kernels.SpectralMixtureKernel(num_mixtures=5, ard_num_dims=2), # bc SM shouldnt be combined with scale kernels
                # TODO set batch size??
                num_dims=2, grid_size=100
            )
            
            self.feature_extractor = feature_extractor
            self.X = train_x
            self.append_sim_column=append_sim_column 

            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1, 1.)
            # was -1 but changed to match the other parts  

        def forward(self, x):
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            self.feature_extractor.joint_gp_training_phase = True
            projected_x = self.feature_extractor(self.X)
            projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
def optimize_acq_DKL(func, m, m1,l,l1, fixed, num_f, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    """Optimize acquisition function."""

    opts = {"maxiter": 200, "maxfun": 200, "disp": False}

    T = 10
    best_value = -999
    best_theta = m1.X[0, :]

    bounds = [(0, 1) for _ in range(m.X.shape[1] - num_f)]
    print(m1.X.shape, m.X.shape)
    for ii in range(T):
        x0 = np.random.uniform(0, 1, m.X.shape[1] - num_f)
        print( 'DHSAPAES', x0, m.X.shape[1] - num_f)


        res = minimize(
            lambda x: -func(m, m1,l,l1, x, fixed),
            x0,
            bounds=bounds,
            method="L-BFGS-B",
            options=opts,
        )

        val = func(m, m1,l,l1, res.x, fixed)
        if val > best_value:
            best_value = val
            best_theta = res.x

    return np.clip(best_theta, 0, 1)


def UCB_DKL(m, m1,l,l1, x, fixed, kappa=None):
    """UCB acquisition function. Interesting points to note:
    1) We concat with the fixed points, because we are not optimizing wrt
       these. This is the Reward and Time, which we can't change. We want
       to find the best hyperparameters *given* the reward and time.
    2) We use m to get the mean and m1 to get the variance. If we already
       have trials running, then m1 contains this information. This reduces
       the variance at points currently running, even if we don't have
       their label.
       Ref: https://jmlr.org/papers/volume15/desautels14a/desautels14a.pdf

    """

    c1 = 0.2
    c2 = 0.4
    beta_t = c1 + max(0, np.log(c2 * m.X.shape[0]))
    kappa = np.sqrt(beta_t) if kappa is None else kappa

    xtest = np.concatenate((fixed.reshape(-1, 1), np.array(x).reshape(-1, 1))).T
    xtest = xtest.astype(np.float32)
    print(xtest)

    try:
        preds = predict(m, l, xtest)#m.predict(xtest)
        mean = preds.mean
        #mean = mean.astype(np.float32)
    except ValueError:
        logger.info('mean is error')
        print('value error in mean, defaulting to -9999')
        mean = -9999

    try:
        preds = predict(m1, l1, xtest)#m1.predict(xtest)
        var = preds.variance
        #var = preds.astype(np.float32)
    except ValueError:
        var = 0
        logger.info('error in varaince')
        print('value error in var, defaulting to 0')
    return mean + kappa * var

def predict(model, likelihood, x):

    model.eval()
    likelihood.eval()
    x = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
        preds = model(x)
    return preds

## KISS-GP wilson et al





def metatrain_DKL_wilson(model, X_train, y_train, likelihood, seed, 
                         training_iterations=100, freeze=False, 
                         warm_start_only=False, X_test=None, y_test=None, 
                         save=None, lr=0.01, ray_tune_exp=False, scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau, batch_size=64):
    
    

    tracemalloc.start()

    # Code block to monitor memory usage    

    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model.train()
    likelihood.train()
    model.feature_extractor.joint_gp_training_phase=True

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters()},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=lr)

    if freeze or warm_start_only:
        # Don't train NN
        optimizer = torch.optim.Adam([
            {'params': model.covar_module.parameters()},
            {'params': model.mean_module.parameters()},
            {'params': model.likelihood.parameters()},
        ], lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    scheduler = scheduler(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if X_test is not None and y_test is not None:
        val_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    logger.info('Training in progress!!')
    iterator = tqdm.tqdm(range(training_iterations))
    loss_values = []
    test_loss_vals = []

    
    for i in iterator:
        train_loss=0.0
        val_rmse=0.0
        val_nnl=0.0
        model.train()
        likelihood.train()
        #for batch_X, batch_y in train_loader:
        
            # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(X_train)
        # Calculate loss and backprop derivatives
        loss = -mll(output, y_train)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()
        # Log the loss value
        train_loss += loss.item()
        train_loss /= len(train_loader)
        loss_values.append(train_loss)
        print('train_loss', train_loss)
        # Evaluate on the test set if provided
        if X_test is not None and y_test is not None:
            model.eval()
            likelihood.eval()
            with torch.no_grad():
                for X_val, y_val in val_dataloader:
                    test_output = model(X_val)
                # Use the predictive distribution to get the mean and variance
                    pred_mean = test_output.mean
                    pred_var = test_output.variance
                # RMSE: Root Mean Squared Error
                    rmse = torch.sqrt(torch.mean((pred_mean - y_val) ** 2)).item() 
                # NLL: Negative Log Likelihood, sum or average over all test points
                    nll = -likelihood.log_marginal(y_val, test_output).mean().item()  # Take the mean NLL
                    val_rmse+=rmse
                    val_nnl+=nll
                
                val_rmse = val_rmse / len(val_dataloader)
                val_nll = val_nnl / len(val_dataloader)
                #scheduler.step(val_nll) #
                test_loss_vals.append(val_rmse)
                #print(f'RMSE: {rmse}, NLL: {nll}')
                print({'train_loss': train_loss, 'val_set_rmse': val_rmse, 'training_itr': i ,'negative_log_likelihood':val_nll})
            # if ray_tune_exp:
            #     train.report({'train_loss': train_loss, 'val_set_rmse': val_rmse, 'training_itr': i ,'negative_log_likelihood':val_nll})


    current, peak = tracemalloc.get_traced_memory()
    curr = round(current / (1024 ** 2) , 2)
    p = round(peak / (1024 ** 2), 2)

    print(f"Current memory usage: {curr} MB")
    print(f"Peak memory usage: {p} MB")
    
    tracemalloc.stop()
    train.report({'peak_mem':p, 'current_mem': curr,'train_loss': train_loss, 'training_itr': i})

    # # Plot and save the training loss if specified
    # if save is not None:
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(loss_values, label='Training Loss')
    #     plt.plot(test_loss_vals, label='val')
    #     plt.title('Training Loss over Iterations')
    #     plt.xlabel('Iteration')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.grid()
    #     plt.savefig('loss.png')
    #     plt.close()

    return model, mll, likelihood