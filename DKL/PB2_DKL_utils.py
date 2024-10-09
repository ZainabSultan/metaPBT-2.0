import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import spearmanr
from torch.optim import SGD, Adam
import torch
from torch.utils.data import DataLoader, TensorDataset
import tqdm.notebook
import gpytorch
from torch.optim.lr_scheduler import MultiStepLR
import torch
import logging
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.optimize import minimize
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)
import random


# def train_neural_network_model(model, X, y, num_epochs=50, batch_size=32, learning_rate=1e-3,  early_stopping_patience=5):
#     # Prepare for cross-validation
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     spearman_scores = []
#     cv_scores = []
    
#     for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
#         print(f"Fold {fold + 1}/{kf.get_n_splits()}")

#         # Split the data
#         X_train, X_val = X[train_idx], X[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]

#         # Create DataLoader
#         train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
#         val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

#         train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#         val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#         # Initialize the model, loss function, and optimizer
#         # input_dim = X.shape[1]
#         # output_dim = 1
#         #model = SimpleNN(input_dim, hidden_dim, output_dim)
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#         # Early stopping variables
#         best_val_loss = float('inf')
#         patience_counter = 0

#         for epoch in range(num_epochs):
#             model.train()
#             running_loss = 0.0
            
#             # Training phase
#             for inputs, targets in train_loader:
#                 optimizer.zero_grad()
#                 outputs = model(inputs)
#                 loss = criterion(outputs.squeeze(), targets)
#                 loss.backward()
#                 optimizer.step()
#                 running_loss += loss.item() * inputs.size(0)
            
#             epoch_loss = running_loss / len(train_loader.dataset)

#             # Validation phase
#             model.eval()
#             val_predictions = []
#             val_targets = []
#             with torch.no_grad():
#                 for inputs, targets in val_loader:
#                     outputs = model(inputs)
#                     val_predictions.append(outputs.squeeze().cpu().numpy())
#                     val_targets.append(targets.cpu().numpy())
            
#             val_predictions = np.concatenate(val_predictions)
#             val_targets = np.concatenate(val_targets)
#             mse = mean_squared_error(val_targets, val_predictions)
#             spearman_corr, _ = spearmanr(val_targets, val_predictions)
            
#             print(f"Epoch {epoch + 1}/{num_epochs}, MSE: {mse:.4f}, Spearman Correlation: {spearman_corr:.4f}")
            
#             # Early stopping check
#             if mse < best_val_loss:
#                 best_val_loss = mse
#                 patience_counter = 0
#             else:
#                 patience_counter += 1
            
#             if patience_counter >= early_stopping_patience:
#                 print("Early stopping triggered.")
#                 break
        
#         spearman_scores.append(spearman_corr)
#         cv_scores.append(mse)
    
#     return spearman_scores, cv_scores



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

    for ii in range(T):
        x0 = np.random.uniform(0, 1, m.X.shape[1] - num_f)

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




def predict(model, likelihood, x):

    model.eval()
    likelihood.eval()
    x = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
        preds = model(x)
    return preds

## KISS-GP wilson et al



def train_DKL_wilson(model, X_train, y_train, likelihood,seed, training_iterations=100, freeze=False):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters()},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=0.01)

    if freeze:
        # dont train NN
        optimizer = torch.optim.Adam([
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=0.01)


    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

#def train():
    logger.info('training in progress!!')
    iterator = tqdm.tqdm(range(training_iterations))
    for i in iterator:
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(X_train)
        # Calc loss and backprop derivatives
        loss = -mll(output, y_train)
        
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()
    
    return model,mll, likelihood

#%time train()






## approximate kernel

# def train_inner_loop(model,optimizer, likelihood, epoch, train_loader):

#     #n_epochs = 1
    
#     #scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)
#     mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))
#     model.train()
#     likelihood.train()

#     minibatch_iter = tqdm.notebook.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
#     with gpytorch.settings.num_likelihood_samples(8):
#         for data, target in minibatch_iter:
#             if torch.cuda.is_available():
#                 data, target = data.cuda(), target.cuda()
#             optimizer.zero_grad()
#             output = model(data)
#             loss = -mll(output, target)
#             loss.backward()
#             optimizer.step()
#             minibatch_iter.set_postfix(loss=loss.item())


# def test():
#     model.eval()
#     likelihood.eval()

#     correct = 0
#     with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
#         for data, target in test_loader:
#             if torch.cuda.is_available():
#                 data, target = data.cuda(), target.cuda()
#             output = likelihood(model(data))  # This gives us 16 samples from the predictive distribution
#             pred = output.probs.mean(0).argmax(-1)  # Taking the mean over all of the sample we've drawn
#             correct += pred.eq(target.view_as(pred)).cpu().sum()
#     print('Test set: Accuracy: {}/{} ({}%)'.format(
#         correct, len(test_loader.dataset), 100. * correct / float(len(test_loader.dataset))
#    ))

# def train_DKL(X, y, model, likelihood, n_epochs):
#     lr = 0.1
#     optimizer = SGD([
#         {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
#         {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
#         {'params': model.gp_layer.variational_parameters()},
#         {'params': likelihood.parameters()},
#     ], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
#     scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)

#     X_train_tensor = torch.tensor(X, dtype=torch.float32)
#     y_train_tensor = torch.tensor(y, dtype=torch.float32)

#     # Step 2: Create a Dataset from tensors
#     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

#     # Step 3: Create a DataLoader
#     batch_size = 32  # You can set the batch size to whatever is appropriate for your task
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


#     for epoch in range(1, n_epochs + 1):
#         with gpytorch.settings.use_toeplitz(False):
#             train_inner_loop(model,optimizer, likelihood, epoch, train_loader)
#             #test()
#         scheduler.step()
#     #state_dict = model.state_dict()
#     #likelihood_state_dict = likelihood.state_dict()
#     #torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, 'dkl_cifar_checkpoint.dat')



# def minimise_wrt_acq(mll, model):
#     # Example training data
    
#     #mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

#     # Fit the model
#     fit_gpytorch_model(mll)

#     # Define the acquisition function
#     UCB = UpperConfidenceBound(model, beta=0.1)

#     # Optimize the acquisition function
#     bounds = torch.stack([torch.zeros(2), torch.ones(2)])
#     candidate, acq_value = optimize_acqf(
#         UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20
#     )
#     return candidate, acq_value



