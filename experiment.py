

from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from models import LinearModel, LinearModelWithGumbel, BaselineMLP, BaselineLinear, LinearModelWithRandomActivations, LinearModelWithDropout
from train import train
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import tqdm
import random
import matplotlib.pyplot as plt

LINEAR_HIDDEN = 64
EPOCHS = 15000
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

def experiment(seed, num_runs=10, dim = 2):
    torch.manual_seed(seed)
    # Generate a 2D classification dataset
    X, y = make_classification(
        n_samples=300,
        n_features=dim,
        n_informative=dim,
        n_redundant=0,
        hypercube=False,
        n_clusters_per_class=2,
        n_classes=2,
        random_state=seed
    )

    # Plot data
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=50)
    plt.title(f"2D Classification Dataset (Seed: {seed})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label="Class")
    plt.grid(True)
    plt.show()

    # Split the dataset into train, validation and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Convert numpy arrays to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Create TensorDataset and DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)


    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Initialize lists to store loss histories for each run
    all_loss_baseline_linear = []
    all_loss_baseline = []
    all_loss_gumbel_baseline = []
    all_loss_gumbel = []
    all_loss_random_activations = []


    for run in tqdm.tqdm(range(num_runs), desc="Runs", leave=False):
        # Train BaselineLinear
        baseline_linear = BaselineLinear(input_dim=dim, output_dim=1).to(device)
        optimizer = torch.optim.Adam(baseline_linear.parameters())
        loss_history_baseline_linear = train(baseline_linear, train_loader, criterion, optimizer, EPOCHS, device=device)
        all_loss_baseline_linear.append(loss_history_baseline_linear)

        # Train BaselineMLP
        baseline_mlp = BaselineMLP(input_dim=dim, output_dim=1, hidden_dim=LINEAR_HIDDEN).to(device)
        optimizer = torch.optim.Adam(baseline_mlp.parameters())
        loss_history_baseline = train(baseline_mlp, train_loader, criterion, optimizer, EPOCHS, device=device)
        all_loss_baseline.append(loss_history_baseline)
        
        activation = torch.nn.Linear(dim, LINEAR_HIDDEN).to(device)
        gumbel_baseline = LinearModelWithGumbel(input_dim=dim, output_dim=1, hidden_dim=LINEAR_HIDDEN, activation_layer=activation, gumbel=True).to(device)

        optimizer = torch.optim.Adam(gumbel_baseline.parameters())
        loss_history_gumbel = train(gumbel_baseline, train_loader, criterion, optimizer, EPOCHS, device=device)
        all_loss_gumbel_baseline.append(loss_history_gumbel)

        gumbel_trained = LinearModelWithGumbel(input_dim=dim, output_dim=1, hidden_dim=LINEAR_HIDDEN, activation_layer=activation, gumbel=False).to(device)
        optimizer = torch.optim.Adam(gumbel_trained.parameters())
        loss_history_gumbel_trained = train(gumbel_trained, train_loader, criterion, optimizer, EPOCHS, device=device)
        all_loss_gumbel.append(loss_history_gumbel_trained)

        random_activations = LinearModelWithRandomActivations(input_dim=dim, output_dim=1, hidden_dim=LINEAR_HIDDEN).to(device)
        optimizer = torch.optim.Adam(random_activations.parameters())
        loss_history_random_activations = train(random_activations, train_loader, criterion, optimizer, EPOCHS, device=device)
        all_loss_random_activations.append(loss_history_random_activations)

    # Store loss histories for each run
    loss_histories = {
        "BaselineLinear": all_loss_baseline_linear,
        "BaselineMLP": all_loss_baseline,
        "GumbelBaseline": all_loss_gumbel_baseline,
        "LinearModelWithGumbel": all_loss_gumbel,
        "LinearModelWithRandomActivations": all_loss_random_activations,
    }



    # Plot training loss curves for both models

    #plt.figure(figsize=(10, 5))

    # BaselineMLP loss curve
    #plt.plot(loss_history_baseline, label='BaselineMLP')

    # LinearModel loss curve
    #plt.plot(loss_history_linear, label='LinearModel')

    # LinearModelWithGumbel loss curve
    #plt.plot(loss_history_gumbel, label='LinearModelWithGumbel')

    #plt.xlabel('Iterations')
    #plt.ylabel('Training Loss')
    #plt.title('Training Loss Curves')
    #plt.legend()
    #plt.grid(True)
    #plt.show()

    # Plot average training loss curves for each model

    # Plot dataset and figures side by side

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot average training loss curves
    
    for model_name, loss_history in loss_histories.items():
        avg_loss = np.mean(loss_history, axis=0)
        axs[0].plot(avg_loss, label=model_name, alpha=0.5)
    axs[0].set_xlabel('Iterations')
    axs[0].set_ylabel('Training Loss')
    axs[0].set_title(f'Average Training Loss Curves N = {num_runs}')
    axs[0].legend()
    axs[0].grid(True)

    # Make scatterplot visualizing the dataset
    # Plot the dataset
    axs[1].scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k')
    axs[1].set_xlabel('Feature 1')
    axs[1].set_ylabel('Feature 2')
    axs[1].set_title('2D Classification Dataset')

    plt.tight_layout()
    plt.show()
    
    
    
    # have the classifiers predict the training set, then visualize the predictions on a scatterplot

    fig, axs = plt.subplots(1, 5, figsize=(35, 5))
    for i, (model_name, model) in enumerate(zip(["BaselineLinear", "BaselineMLP", "gumbel_baseline", "gumbel_trained", "random_activations"], [baseline_linear, baseline_mlp, gumbel_baseline, gumbel_trained, random_activations])):
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train_tensor).sigmoid()
            pred_labels = (y_pred > 0.5).float().squeeze().cpu().numpy()
            true_labels = y_train_tensor.squeeze().cpu().numpy()
            correct = pred_labels == true_labels

            # Plot correct predictions
            axs[i].scatter(
                X_train[correct, 0], X_train[correct, 1],
                c='green', label='Correct', edgecolor='k'
            )
            # Plot incorrect predictions
            axs[i].scatter(
                X_train[~correct, 0], X_train[~correct, 1],
                c='orange', label='Incorrect', edgecolor='k'
            )
            
            # Plot decision boundary
            xx, yy = np.meshgrid(
                np.linspace(X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5, 200),
                np.linspace(X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5, 200)
            )
            grid = np.c_[xx.ravel(), yy.ravel()]
            grid_tensor = torch.tensor(grid, dtype=torch.float32)
            with torch.no_grad():
                zz = model(grid_tensor).sigmoid().cpu().numpy().reshape(xx.shape)
            axs[i].contourf(xx, yy, zz, levels=[0, 0.5, 1], alpha=0.2, colors=['red', 'blue'])

        axs[i].set_xlabel('Feature 1')
        axs[i].set_ylabel('Feature 2')
        axs[i].set_title(f'Predictions: {model_name}')
        axs[i].legend()


    plt.show()


experiment(seed=533, num_runs=5)


# LR, width, 

parameters = {"learning_rate": [0.001, 0.01, 0.05], "width": [1024, 2048, 4096]}


def param_search(params, model_class, dim, train_loader, criterion, num_runs=1, seed=42, device='cpu'):
    best_model = None
    best_params = None
    best_loss = float('inf')
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    for lr in params['learning_rate']:
        for width in params['width']:
            avg_losses = []
            for _ in range(num_runs):
                
                if model_class == LinearModelWithGumbel:
                    activation_best_loss = float('inf')
                    activation_best_model = None
                    activation_best_layer = None
                    activation_best_params = None
                    for lr_ in params['learning_rate']:
                        activation_layer = torch.nn.Linear(dim, width).to(device)
                        model = model_class(input_dim=dim, output_dim=1, hidden_dim=width, activation_layer=activation_layer, gumbel=True).to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr_)
                        loss_history = train(model, train_loader, criterion, optimizer, EPOCHS, device=device)
                        avg_loss = np.mean(loss_history[-100:])  # Check last 100 epochs
                        if avg_loss < activation_best_loss:
                            activation_best_loss = avg_loss
                            activation_best_model = model
                            activation_best_layer = activation_layer
                            activation_best_params = {'learning_rate': lr_, 'width': width}
                            print("activation params", activation_best_params)
                    model = model_class(input_dim=dim, output_dim=1, hidden_dim=width, activation_layer=activation_best_layer, gumbel=False).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    loss_history = train(model, train_loader, criterion, optimizer, EPOCHS, device=device)

                else:
                    model = model_class(input_dim=dim, hidden_dim=width, output_dim=1).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    loss_history = train(model, train_loader, criterion, optimizer, EPOCHS, device=device)

                avg_loss = np.mean(loss_history[-100:])  # Check last 100 epochs
                avg_losses.append(avg_loss)
            mean_loss = np.mean(avg_losses)
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_model = model
                if model_class == LinearModelWithGumbel:
                    best_params = {'learning_rate': lr, 'width': width, 'activation_layer': activation_best_layer, 'activation_params': activation_best_params}
                else:
                    best_params = {'learning_rate': lr, 'width': width}
                print("best params", best_params)


    return best_model, best_params



def find_best_models(num_runs, seed, parameters):
    
    # Create dataset
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    X, y = make_classification(
        n_samples=300,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        hypercube=False,
        n_clusters_per_class=2,
        n_classes=2,
        random_state=seed
    )
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=50)
    plt.title(f"2D Classification Dataset (Seed: {seed})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label="Class")
    plt.grid(True)
    plt.show()

    train_loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)), batch_size=32, shuffle=True)
    criterion = torch.nn.BCEWithLogitsLoss()
    results = {}
    for model_class in [BaselineMLP, LinearModelWithGumbel, LinearModelWithRandomActivations]:
        print(f"Finding best params for {model_class.__name__}")
        best_model, best_params = param_search(parameters, model_class, dim=2, train_loader=train_loader, criterion=criterion, num_runs=num_runs, seed=seed, device=device)
        results[model_class.__name__] = {
            "best_model": best_model,
            "best_params": best_params
        }
        print(f"Best params for {model_class.__name__}: {best_params}")

    for model_name, result in results.items():
        print(f"Results for {model_name}:")
        print(f"  Best Params: {result['best_params']}")
        print(f"  Best Model: {result['best_model']}")

