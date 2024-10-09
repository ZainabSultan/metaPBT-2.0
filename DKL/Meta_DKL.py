import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn
from ray.tune import TuneError
from ray.tune.experiment import Trial
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pbt import _PBTTrialState
from ray.tune.utils.util import flatten_dict, unflatten_dict
from ray.util.debug import log_once
#from DKL.PB2_DKL_utils import train_DKL_wilson, UCB_DKL,optimize_acq_DKL
from DKL.Meta_DKL_data_utils import process_pb2_runs_metadata, standardize, normalize
from DKL.Meta_DKL_train_utils import metatrain_DKL_wilson, UCB_DKL, optimize_acq_DKL, DynamicNN, GPRegressionModel_DKL
import math
import torch
import gpytorch
import random
logging.getLogger().setLevel(logging.INFO)
if TYPE_CHECKING:
    from ray.tune.execution.tune_controller import TuneController


def import_pb2_dependencies():
    try:
        import GPy
    except ImportError:
        GPy = None
    try:
        import sklearn
    except ImportError:
        sklearn = None
    return GPy, sklearn


GPy, has_sklearn = import_pb2_dependencies()

if GPy and has_sklearn:

    from DKL.pb2_utils import get_limits

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import random
import numpy as np






def _fill_config(
    config: Dict, hyperparam_bounds: Dict[str, Union[dict, list, tuple]],seed
) -> Dict:
    """Fills missing hyperparameters in config by sampling uniformly from the
    specified `hyperparam_bounds`.
    Recursively fills the config if `hyperparam_bounds` is a nested dict.

    This is a helper used to set initial hyperparameter values if the user doesn't
    specify them in the Tuner `param_space`.

    Returns the dict of filled hyperparameters.
    """
    #np.random.seed(seed)
    filled_hyperparams = {}
    for param_name, bounds in hyperparam_bounds.items():
        if isinstance(bounds, dict):
            if param_name not in config:
                config[param_name] = {}
            filled_hyperparams[param_name] = _fill_config(config[param_name], bounds,seed)
        elif isinstance(bounds, (list, tuple)) and param_name not in config:
            if log_once(param_name + "-missing"):
                logger.debug(
                    f"Cannot find {param_name} in config. Initializing by "
                    "sampling uniformly from the provided `hyperparam_bounds`."
                )
            assert len(bounds) == 2
            low, high = bounds
            config[param_name] = filled_hyperparams[param_name] = np.random.uniform(
                low, high
            )
    return filled_hyperparams



def _select_config(
    self,
    Xraw: np.array,
    yraw: np.array,
    current: list,
    newpoint: np.array,

) -> np.ndarray:
    """Selects the next hyperparameter config to try.

    This function takes the formatted data, fits the GP model and optimizes the
    UCB acquisition function to select the next point.

    Args:
        Xraw: The un-normalized array of hyperparams, Time and
            Reward - reward is for example the reward of an rl agent
        yraw: The un-normalized vector of reward changes.
        current: The hyperparams of trials currently running. This is
            important so we do not select the same config twice. If there is
            data here then we fit a second GP including it
            (with fake y labels). The GP variance doesn't depend on the y
            labels so it is ok.
        newpoint: The Reward and Time for the new point.
            We cannot change these as they are based on the *new weights*.
        bounds: Bounds for the hyperparameters. Used to normalize.
        num_f: The number of fixed params. Almost always 2 (reward+time)

    Return:
        xt: A vector of new hyperparameters.
    """
    # for DKL
    bounds = self._hyperparam_bounds_flat
    neural_network = self.neural_network
    seed = self.seed
    if np.isscalar(self.sim_feature):
        # If sim_feature is a scalar, reshape it to be a single column
        sim_feature_col = np.full((Xraw.shape[0], 1), self.sim_feature)
    else:
        # If sim_feature is an array, create multiple columns (one per element in the array)
        sim_feature_col = np.tile(self.sim_feature, (Xraw.shape[0], 1))

    # Insert sim_feature as the third column (after the first two columns of Xraw)
    
    Xraw_ = np.hstack([Xraw[:, :2], sim_feature_col, Xraw[:, 2:]])
    print(Xraw.shape, Xraw_.shape, self.metadata_train_x.values.shape)
    # oldpoints = Xraw[:, :self.num_fixed_params ]
    #
    # limits = get_limits(oldpoints, bounds)    

    # length = select_length(Xraw, yraw, bounds, num_f)

    # Xraw = Xraw[-length:, :]
    # yraw = yraw[-length:]

    X_train = np.concatenate([self.metadata_train_x.values, Xraw_], axis=0)
    base_vals = np.array(list(bounds.values())).T
    oldpoints = X_train[:, :self.num_fixed_params ]
    # old_lims = np.concatenate(
    #     (np.max(oldpoints, axis=0), np.min(oldpoints, axis=0))
    # ).reshape(2, oldpoints.shape[1])
    # limits = np.concatenate((old_lims, base_vals), axis=1)
    limits = get_limits(oldpoints, bounds)
    X = normalize(X_train, limits)
    y = standardize(yraw)
    y = np.concatenate([self.metadata_train_y.values, y], axis=0)
    newpoint_ = np.hstack([newpoint[:2], self.sim_feature, newpoint[ 2:]])
    print(newpoint_, oldpoints.shape)
    fixed = normalize(newpoint_, oldpoints)
    
    logger.info('about to go train')
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    train_x = torch.tensor(X, dtype=torch.float32)
    train_y =torch.squeeze(torch.tensor(y, dtype=torch.float32))
    print(X.shape, y.shape)
    
    m = self.deep_kernel_gaussian_model(train_x, train_y, feature_extractor=neural_network, likelihood=likelihood,seed=seed)
    m_trained, mll_m ,l= metatrain_DKL_wilson(model=m, X_train=train_x, y_train=train_y, likelihood=likelihood,seed=seed)
    # if there are current runs you must freeze the neural network heart in order 
    # not to corrupt it with the fake values
    if current is None:
        m1_trained = deepcopy(m_trained)
        l1 = deepcopy(l)
    else:
        # add the current trials to the dataset
        logger.info('training new gp')
        fixed= fixed.astype(np.float32)
        padding = np.array([fixed for _ in range(current.shape[0])])
        current = normalize(current, base_vals)
        current = np.hstack((padding, current))
        Xnew = np.vstack((X, current))
        # fake labels bc we dont depend on y for variance calculation
        ypad = np.zeros(current.shape[0])
        ypad = ypad.reshape(-1, 1)
        ynew = np.vstack((y, ypad))

        train_xnew = torch.tensor(Xnew, dtype=torch.float32)
        train_ynew = torch.squeeze(torch.tensor(ynew, dtype=torch.float32))
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        m1 = self.deep_kernel_gaussian_model(train_xnew, train_ynew, feature_extractor=neural_network, likelihood=likelihood,seed=seed)
        m1_trained, mll_m1,l1 = metatrain_DKL_wilson(model=m1, X_train=train_xnew, y_train=train_ynew, likelihood=likelihood, freeze=True,seed=seed)

    #xt = minimise_wrt_acq()
    xt = optimize_acq_DKL(UCB_DKL, m_trained, m1_trained,l,l1, fixed, self.num_fixed_params ,seed)

    # convert back...denormalise
    xt = xt * (np.max(base_vals, axis=0) - np.min(base_vals, axis=0)) + np.min(
        base_vals, axis=0
    )

    xt = xt.astype(np.float32)
    return xt



def _explore(
    self,
    base: Trial,
    old: Trial,
    config: Dict[str, Tuple[float, float]]
) -> Tuple[Dict, pd.DataFrame]:
    """Returns next hyperparameter configuration to use.

    This function primarily processes the data from completed trials
    and then requests the next config from the select_config function.
    It then adds the new trial to the dataframe, so that the reward change
    can be computed using the new weights.
    It returns the new point and the dataframe with the new entry.
    """
    data = self.data
    bounds = self._hyperparam_bounds_flat
    current = self.current


    df = data.sort_values(by="Time").reset_index(drop=True)

    # Group by trial ID and hyperparams.
    # Compute change in timesteps and reward.
    df["y"] = df.groupby(["Trial"] + list(bounds.keys()))["Reward"].diff()
    df["t_change"] = df.groupby(["Trial"] + list(bounds.keys()))["Time"].diff()

    # Delete entries without positive change in t.
    df = df[df["t_change"] > 0].reset_index(drop=True)
    df["R_before"] = df.Reward - df.y

    # Normalize the reward change by the update size.
    # For example if trials took diff lengths of time.
    df["y"] = df.y / df.t_change
    df = df[~df.y.isna()].reset_index(drop=True)
    df = df.sort_values(by="Time").reset_index(drop=True)

    # Only use the last 1k datapoints, so the GP is not too slow.
    df = df.iloc[-1000:, :].reset_index(drop=True)

    # We need this to know the T and Reward for the weights.
    dfnewpoint = df[df["Trial"] == str(base)]

    if not dfnewpoint.empty:
        # N ow specify the dataset for the GP.
        y = np.array(df.y.values)
        # Meta data we keep -> episodes and reward.
        # (TODO: convert to curve)
        t_r = df[["Time", "R_before"]]
        hparams = df[bounds.keys()]
        X = pd.concat([t_r, hparams], axis=1).values
        newpoint = df[df["Trial"] == str(base)].iloc[-1, :][["Time", "R_before"]].values
        new = _select_config(self, X, y, current, newpoint)

        new_config = config.copy()
        values = []
        # Cast types for new hyperparameters.
        for i, col in enumerate(hparams.columns):
            # Use the type from the old config. Like this types
            # should be passed on from the first config downwards.
            type_ = type(config[col])
            new_config[col] = type_(new[i])
            values.append(type_(new[i]))

        new_T = df[df["Trial"] == str(base)].iloc[-1, :]["Time"]
        new_Reward = df[df["Trial"] == str(base)].iloc[-1, :].Reward

        lst = [[str(old)] + [new_T] + values + [new_Reward]]
        cols = ["Trial", "Time"] + list(bounds) + ["Reward"]
        new_entry = pd.DataFrame(lst, columns=cols)

        # Create an entry for the new config, with the reward from the
        # copied agent.
        data = pd.concat([data, new_entry]).reset_index(drop=True)

    else:
        new_config = config.copy()

    return new_config, data




class Meta_DKL(PopulationBasedTraining):

    def __init__(
        self,
        time_attr: str = "time_total_s",
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        perturbation_interval: float = 60.0,
        hyperparam_bounds: Dict[str, Union[dict, list, tuple]] = None,
        quantile_fraction: float = 0.25,
        log_config: bool = True,
        require_attrs: bool = True,
        synch: bool = False,
        custom_explore_fn: Optional[Callable[[dict], dict]] = None,
        seed:int =None,
        meta_data_dir_list: list=None,
        current_env_config: dict =None,
        num_extra_info_dims: int = 1,
        warmstart_only: bool = False,
        sim_feature: float = 0.0,
        type_meta_runs: str = 'pb2',
        deep_kernel_gaussian_model = GPRegressionModel_DKL,
        pretrain_neural_net: bool = True,
        neural_net_archi: list = [
        {'type': 'linear', 'out_dim': 1000},
        {'type': 'relu'},
        {'type': 'linear', 'out_dim': 500},
        {'type': 'relu'},
        {'type': 'linear', 'out_dim': 50},
        {'type': 'relu'},
        {'type': 'linear', 'out_dim': 2}  # Feature extraction output size
    ]
        
            ):

        gpy_available, sklearn_available = import_pb2_dependencies()
        if not gpy_available:
            raise RuntimeError("Please install GPy to use PB2.")

        if not sklearn_available:
            raise RuntimeError("Please install scikit-learn to use PB2.")

        hyperparam_bounds = hyperparam_bounds or {}

        if not hyperparam_bounds:
            raise TuneError(
                "`hyperparam_bounds` must be specified to use PB2 scheduler."
            )
        
        if not meta_data_dir_list:
            raise TuneError(
                "`meta dir list needs to be specified to do meta learning."
            )


        super(Meta_DKL, self).__init__(
            time_attr=time_attr,
            metric=metric,
            mode=mode,
            perturbation_interval=perturbation_interval,
            hyperparam_mutations=hyperparam_bounds,
            quantile_fraction=quantile_fraction,
            resample_probability=0,
            custom_explore_fn=custom_explore_fn,
            log_config=log_config,
            require_attrs=require_attrs,
            synch=synch,
            
        )

        self.last_exploration_time = 0  # when we last explored
        self.data = pd.DataFrame()
        self.pretrain_neural_net = pretrain_neural_net
        self.neural_net_archi= neural_net_archi
        self.seed=seed
        self.meta_data_dir_list=meta_data_dir_list
        self.current_env_config = current_env_config
        self._hyperparam_bounds = hyperparam_bounds
        self._hyperparam_bounds_flat = flatten_dict(
            hyperparam_bounds, prevent_delimiter=True
        )
        self.sim_feature = sim_feature
        self.warmstart_only = warmstart_only
        self._validate_hyperparam_bounds(self._hyperparam_bounds_flat)
        self.deep_kernel_gaussian_model = deep_kernel_gaussian_model

        # Current = trials running that have already re-started after reaching
        #           the checkpoint. When exploring we care if these trials
        #           are already in or scheduled to be in the next round.
        self.current = None
        self.num_fixed_params = 2 + num_extra_info_dims # 2 = time and reward

        num_inputs = self.num_fixed_params + len(hyperparam_bounds) # time, reward, ,sim feat and num of HPs
        self.neural_network = DynamicNN(self.neural_net_archi,input_dim=num_inputs, seed=seed) 

        if type_meta_runs == 'pb2':

            x_train, y_train, x_val, y_val = process_pb2_runs_metadata(self.meta_data_dir_list, self._hyperparam_bounds, current_env_config=self.current_env_config)
        else:
            ...

        self.metadata_train_x = x_train
        self.metadata_train_y = y_train
        self.metadata_val_x = x_val
        self.metadata_val_y = y_val
        
        # if self.pretrain_neural_net:
        #     self.neural_network = pretrain_neural_network_model(model=self.neural_network, X_train=x_train,y_train=y_train, X_val=x_val,y_val= y_val ,seed=seed, num_epochs=100)

    def on_trial_add(self, tune_controller: "TuneController", trial: Trial):
        filled_hyperparams = _fill_config(trial.config, self._hyperparam_bounds,self.seed)
        # Make sure that the params we sampled show up in the CLI output
        trial.evaluated_params.update(flatten_dict(filled_hyperparams))
        super().on_trial_add(tune_controller, trial)

    def _validate_hyperparam_bounds(self, hyperparam_bounds: dict):
        """Check that each hyperparam bound is of the form [low, high].

        Raises:
            ValueError: if any of the hyperparam bounds are of an invalid format.
        """
        for key, value in hyperparam_bounds.items():
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(
                    "`hyperparam_bounds` values must either be "
                    f"a list or tuple of size 2, but got {value} "
                    f"instead for the param '{key}'"
                )
            low, high = value
            if low > high:
                raise ValueError(
                    "`hyperparam_bounds` values must be of the form [low, high] "
                    f"where low <= high, but got {value} instead for param '{key}'."
                )

    def _save_trial_state(
        self, state: _PBTTrialState, time: int, result: Dict, trial: Trial
    ):
        score = super(Meta_DKL, self)._save_trial_state(state, time, result, trial)

        # Data logging for PB2.

        # Collect hyperparams names and current values for this trial.
        names = list(self._hyperparam_bounds_flat.keys())
        flattened_config = flatten_dict(trial.config)
        values = [flattened_config[key] for key in names]

        # Store trial state and hyperparams in dataframe.
        # this needs to be made more general.
        lst = [[trial, result[self._time_attr]] + values + [score]]
        cols = ["Trial", "Time"] + names + ["Reward"]
        entry = pd.DataFrame(lst, columns=cols)

        self.data = pd.concat([self.data, entry]).reset_index(drop=True)
        self.data.Trial = self.data.Trial.astype("str")

    def _get_new_config(self, trial: Trial, trial_to_clone: Trial) -> Tuple[Dict, Dict]:
        """Gets new config for trial by exploring trial_to_clone's config using
        Bayesian Optimization (BO) to choose the hyperparameter values to explore.

        Overrides `PopulationBasedTraining._get_new_config`.

        Args:
            trial: The current trial that decided to exploit trial_to_clone.
            trial_to_clone: The top-performing trial with a hyperparameter config
                that the current trial will explore.

        Returns:
            new_config: New hyperparameter configuration (after BO).
            operations: Empty dict since PB2 doesn't explore in easily labeled ways
                like PBT does.
        """
        # If we are at a new timestep, we dont want to penalise for trials
        # still going.
        if self.data["Time"].max() > self.last_exploration_time:
            self.current = None

        new_config_flat, data = _explore(
            # self.data,
            # self._hyperparam_bounds_flat,
            # self.current,
            self,
            trial_to_clone,
            trial,
            flatten_dict(trial_to_clone.config),
            #self.neural_network,
            #seed=self.seed
        )

        # Important to replace the old values, since we are copying across
        self.data = data.copy()

        # If the current guy being selecting is at a point that is already
        # done, then append the data to the "current" which contains the
        # points in the current batch.
        new = [new_config_flat[key] for key in self._hyperparam_bounds_flat]

        new = np.array(new)
        new = new.reshape(1, new.size)
        if self.data["Time"].max() > self.last_exploration_time:
            self.last_exploration_time = self.data["Time"].max()
            self.current = new.copy()
        else:
            self.current = np.concatenate((self.current, new), axis=0)
            logger.debug(self.current)

        new_config = unflatten_dict(new_config_flat)

        if self._custom_explore_fn:
            new_config = self._custom_explore_fn(new_config)
            assert (
                new_config is not None
            ), "Custom explore function failed to return a new config"

        return new_config, {}





