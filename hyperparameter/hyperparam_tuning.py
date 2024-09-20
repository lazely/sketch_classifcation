import optuna
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import yaml
from optuna.samplers import TPESampler
from src.config import *
from scripts import train

config = get_config()

hp_config = config['hyperparameter_optimization']

def objective(trial):
    params = {}

    for param, settings in hp_config['parameters'].items():
        if settings['type'] == 'log_uniform':
            params[param] = trial.suggest_loguniform(param, float(settings['min']), float(settings['max']))
        elif settings['type'] == 'uniform':
            params[param] = trial.suggest_uniform(param, settings['min'], settings['max'])
        elif settings['type'] == 'int':
            params[param] = trial.suggest_int(param, settings['min'], settings['max'])
        elif settings['type'] == 'categorical':
            params[param] = trial.suggest_categorical(param, settings['values'])

    best_val_metric = train.main(params, trial_number=trial.number)
    return best_val_metric

def optimize_hyperparameters():
    study = optuna.create_study(direction="maximize", sampler=TPESampler())
    study.optimize(objective, n_trials=hp_config['n_trials'])
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("   {}: {}".format(key, value))
    return trial.params

if __name__ == "__main__":
    best_params = optimize_hyperparameters()