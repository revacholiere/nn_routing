import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import numpy as np
import torch
from typing import Optional, Dict, Any
from tqdm import tqdm
import random
from models import ShallowModel, BaselineMLP, DeepModel
from train import train
from torch.utils.data import TensorDataset, DataLoader
from itertools import product

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

covertype = sklearn.datasets.fetch_covtype(random_state=42, shuffle=True)
X = covertype.data[:50000]
y = covertype.target[:50000]
X = preprocessing.StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Total samples: {len(X)}")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

import torch.nn as nn
import torch.optim as optim

def cross_validate_torch_model(
    model_class,
    X,
    y,
    n_splits=5,
    epochs=10,
    batch_size=128,
    lr=1e-3,
    device='cpu',
    model_params: Optional[Dict[str, Any]] = None,
    early_stopping: bool = False,
    patience: int = 5,
    min_delta: float = 1e-4,
    seed: int = 42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    losses = []
    train_loss_history = []
    val_loss_history = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train-1, dtype=torch.long).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val-1, dtype=torch.long).to(device)

        params = model_params.copy() if model_params else {}
        params.update({'input_dim': X.shape[1], 'output_dim': len(np.unique(y))})
        model = model_class(**params).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_loss = float('inf')
        patience_counter = 0

        fold_train_losses = []
        fold_val_losses = []

        epoch_bar = tqdm(range(epochs), desc=f"Fold {fold}/{n_splits}", leave=False)
        for epoch in epoch_bar:
            model.train()
            permutation = torch.randperm(X_train_tensor.size(0))
            batch_losses = []
            for i in range(0, X_train_tensor.size(0), batch_size):
                indices = permutation[i:i+batch_size]
                batch_x = X_train_tensor[indices]
                batch_y = y_train_tensor[indices]

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            avg_train_loss = np.mean(batch_losses)
            fold_train_losses.append(avg_train_loss)

            # Compute validation loss for this epoch
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = log_loss(
                    y_val_tensor.cpu().numpy(),
                    val_outputs.softmax(dim=1).cpu().numpy()
                )
            fold_val_losses.append(val_loss)
            epoch_bar.set_postfix({'train_loss': avg_train_loss, 'val_loss': val_loss})

            if early_stopping:
                if best_loss - val_loss > min_delta:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    break

        model.eval()
        with torch.no_grad():
            outputs = model(X_val_tensor)
            val_loss = log_loss(y_val_tensor.cpu().numpy(), outputs.softmax(dim=1).cpu().numpy())
            losses.append(val_loss)
        train_loss_history.append(fold_train_losses)
        val_loss_history.append(fold_val_losses)
        #print(f"Fold {fold} val_loss: {val_loss:.4f}")
    return np.mean(losses), train_loss_history, val_loss_history
def hyperparameter_search(
    model_class,
    X,
    y,
    param_grid,
    n_splits=5,
    epochs=100,
    batch_size=128,
    device='cpu',
    early_stopping=True,
    patience=10,
    min_delta=1e-4,
    num_runs=5,
    base_seed=42
):
    keys = list(param_grid.keys())
    best_loss = float('inf')
    best_params = None
    results = []

    param_combinations = list(product(*param_grid.values()))
    outer_bar = tqdm(param_combinations, desc="Hyperparameter grid", leave=True)
    for values in outer_bar:
        params = dict(zip(keys, values))
        losses = []
        epochs_used = []
        outer_bar.set_postfix(params)
        print(f"Testing params: {params}")
        run_bar = tqdm(range(num_runs), desc="Runs", leave=False)
        for run in run_bar:
            seed = base_seed + run
            lr = params.get('lr', 1e-3)
            model_params = params.copy()
            model_params.pop('lr', None)
            avg_loss, train_hist, val_hist = cross_validate_torch_model(
                model_class,
                X,
                y,
                n_splits=n_splits,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
                model_params=model_params,
                early_stopping=early_stopping,
                patience=patience,
                min_delta=min_delta,
                seed=seed
            )
            losses.append(avg_loss)
            # Save the number of epochs actually run (per fold, take the max for each run)
            epochs_this_run = [len(fold) for fold in val_hist]
            epochs_used.append(max(epochs_this_run) if epochs_this_run else 0)
            run_bar.set_postfix({'avg_loss': avg_loss})
        mean_loss = np.mean(losses)
        max_epochs = max(epochs_used)
        results.append((params, mean_loss, max_epochs))
        params.update({'epochs': max_epochs})
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_params = params.copy()
        print(f"  Avg val loss over {num_runs} runs: {mean_loss:.4f}, Max epochs: {max_epochs:.1f}")

    print(f"Best params: {best_params} with loss {best_loss:.4f}")
    return best_params, best_loss, results



def plot_loss_histories(train_loss_histories, val_loss_histories):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Plot training loss
    for model_name, histories in train_loss_histories.items():
        # Truncate all histories to the minimum length
        #print(len(histories))
        #print([len(h) for h in histories])
        min_len = min([len(h) for h in histories])
        histories = [h[:min_len] for h in histories]
        histories = np.array(histories)
        mean_loss = histories.mean(axis=0)
        stderr_loss = histories.std(axis=0) / np.sqrt(histories.shape[0])
        epochs = np.arange(1, len(mean_loss) + 1)
        axs[0].plot(epochs, mean_loss, label=model_name)
        axs[0].fill_between(epochs, mean_loss - stderr_loss, mean_loss + stderr_loss, alpha=0.2)
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Train Loss")
    axs[0].set_title("Training Loss History (mean ± stderr)")
    axs[0].legend()
    axs[0].grid(True)

    # Plot validation loss
    for model_name, histories in val_loss_histories.items():
        min_len = min([len(h) for h in histories])
        histories = [h[:min_len] for h in histories]
        histories = np.array(histories)
        mean_loss = histories.mean(axis=0)
        stderr_loss = histories.std(axis=0) / np.sqrt(histories.shape[0])
        epochs = np.arange(1, len(mean_loss) + 1)
        axs[1].plot(epochs, mean_loss, label=model_name)
        axs[1].fill_between(epochs, mean_loss - stderr_loss, mean_loss + stderr_loss, alpha=0.2)
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Validation Loss")
    axs[1].set_title("Validation Loss History (mean ± stderr)")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
    
    


param_grid = {
    'lr': [1e-2, 1e-3],
    'num_layers': [1, 2],
    'hidden_dim': [64, 128, 256]
}
'''
print("Baseline")
baseline_best_params, baseline_best_loss, baseline_results = hyperparameter_search(
    BaselineMLP,
    X_train,
    y_train,
    param_grid,
    n_splits=5,
    epochs=100,
    batch_size=128,
    device='cpu',
    early_stopping=True,
    patience=10,
    min_delta=1e-4,
    num_runs=5,
    base_seed=42
)

print("Ours")
ours_best_params, ours_best_loss, ours_results = hyperparameter_search(
    ShallowModel,
    X_train,
    y_train,
    param_grid,
    n_splits=5,
    epochs=100,
    batch_size=128,
    device='cpu',
    early_stopping=True,
    patience=10,
    min_delta=1e-4,
    num_runs=5,
    base_seed=42
)

baseline_val_loss, baseline_train_history, baseline_val_history = cross_validate_torch_model(
    BaselineMLP,
    X_train,
    y_train,
    n_splits=5,
    epochs=baseline_best_params['epochs'],
    batch_size=128,
    device='cpu',
    early_stopping=True,
    patience=10,
    min_delta=1e-4,
    model_params=baseline_best_params,
    lr=baseline_best_params['lr']
)

ours_best_val_loss, ours_train_history, ours_val_history = cross_validate_torch_model(
    ShallowModel,
    X_train,
    y_train,
    n_splits=5,
    epochs=ours_best_params['epochs'],
    batch_size=128,
    device='cpu',
    early_stopping=True,
    patience=10,
    min_delta=1e-4,
    model_params=ours_best_params,
    lr=ours_best_params['lr']
)

plot_loss_histories(
    {'Baseline MLP': baseline_train_history},
    {'Baseline MLP': baseline_val_history}
)
'''
baseline_params = {'lr':0.001, 'num_layers':1, 'hidden_dim':64}
ours_params = {'lr':0.001, 'num_layers':1, 'hidden_dim':64}
'''

baseline_val_loss, baseline_train_history, baseline_val_history = cross_validate_torch_model(
    BaselineMLP,
    X_train,
    y_train,
    n_splits=5,
    epochs=200,
    batch_size=128,
    device='cpu',
    early_stopping=False,
    patience=10,
    min_delta=1e-4,
    model_params=baseline_params,
    lr=baseline_lr
)

ours_best_val_loss, ours_train_history, ours_val_history = cross_validate_torch_model(
    ShallowModel,
    X_train,
    y_train,
    n_splits=5,
    epochs=200,
    batch_size=128,
    device='cpu',
    early_stopping=False,
    patience=10,
    min_delta=1e-4,
    model_params=ours_params,
    lr=ours_lr
)
'''


def compare_models(model_class_list, param_list, seed, num_runs):
    all_train_histories = {}
    all_val_histories = {}
    for model, params in zip(model_class_list, param_list):
        print(f"Training {model.__name__} with params: {params}")
        lr = params['lr']
        model_params = params.copy()
        model_params.pop('lr')
        epochs = model_params.pop('epochs', 50)  # Default to 50 if not specified
        train_histories = []
        val_histories = []
        for run in range(num_runs):
            run_seed = seed + run
            avg_val_loss, train_history, val_history = cross_validate_torch_model(
                model,
                X_train,
                y_train,
                n_splits=5,
                epochs=epochs,
                batch_size=128,
                device=device,
                early_stopping=False,
                patience=10,
                min_delta=1e-4,
                model_params=model_params,
                lr=lr,
                seed=run_seed
            )
            train_histories.extend(train_history)
            val_histories.extend(val_history)
            print(f"Run {run+1}/{num_runs} - Avg val loss: {avg_val_loss:.4f}")
        all_train_histories[model.__name__] = train_histories
        all_val_histories[model.__name__] = val_histories
    plot_loss_histories(all_train_histories, all_val_histories)


#compare_models([BaselineMLP, ShallowModel, DeepModel], [baseline_params, ours_params, ours_params], seed=42, num_runs=5)

def tune_models(model_class_list, param_grid, seed, num_runs):
    all_best_params = {}
    all_best_losses = {}
    all_results = {}
    
    for model in model_class_list:
        best_params, best_loss, results = hyperparameter_search(
            model,
            X_train,
            y_train,
            param_grid,
            n_splits=5,
            epochs=200,
            batch_size=128,
            device='cpu',
            early_stopping=True,
            patience=10,
            min_delta=1e-4,
            num_runs=num_runs,
            base_seed=seed
        )
        all_best_params[model.__name__] = best_params
        all_best_losses[model.__name__] = best_loss
        all_results[model.__name__] = results
    return all_best_params, all_best_losses, all_results


#best_params, best_losses, results = tune_models([BaselineMLP, ShallowModel, DeepModel], param_grid, seed=42, num_runs=5)
#compare_models([BaselineMLP, ShallowModel, DeepModel], [best_params['BaselineMLP'], best_params['ShallowModel'], best_params['DeepModel']], seed=42, num_runs=5)

baseline_best_params = {'lr': 0.001, 'num_layers': 2, 'hidden_dim': 256, 'epochs': 63}
ours_shallow_best_params = {'lr': 0.001, 'num_layers': 1, 'hidden_dim': 256, 'epochs': 98}
ours_deep_best_params = {'lr': 0.001, 'num_layers': 1, 'hidden_dim': 128, 'epochs': 59}

compare_models([BaselineMLP, ShallowModel, DeepModel], [baseline_best_params, ours_shallow_best_params, ours_deep_best_params], seed=42, num_runs=5)
