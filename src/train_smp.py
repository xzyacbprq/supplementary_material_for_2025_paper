import sys
sys.path.append('libs/smp')
# Ignore warnings to avoid unnecessary clutter
import warnings
warnings.filterwarnings("ignore")

# Standard libraries for file handling and system operations
import os
import time
import random
import shutil
from pathlib import Path
import glob

# Data manipulation and scientific libraries
import numpy as np
import pandas as pd
from tqdm import tqdm  # For progress bars
import matplotlib.pyplot as plt
#import seaborn as sns
import matplotlib.colors  # For custom color maps

# PyTorch and related libraries for model building, training, and evaluation
import torch
from torch import nn
import torchmetrics
from torch.optim import Adam
import torch.nn.functional as F
#from torchviz import make_dot # For Plotting Model
#from torchsummary import summary  # For model summary
import torchvision.transforms.functional as TF  # For image transforms
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp  # For segmentation models

# Image processing libraries for augmentation and transformations
import cv2
import albumentations as A  # For advanced augmentations

# Set random seed for reproducibility
np.random.seed(42)

def train(model, train_loader, val_loader, epochs, lr, loss_fn, regularization=None, reg_lambda=None, patience=None, verbose=False, model_title="Model", save=False, device = 'cuda'):
    """
    Training loop with early stopping and optional regularization for a PyTorch model.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        epochs (int): Number of epochs to train.
        lr (float): Learning rate for optimizer.
        loss_fn (function): Loss function to use.
        regularization (str, optional): Type of regularization ('L1' or 'L2'). Default is None.
        reg_lambda (float, optional): Regularization coefficient. Default is None.
        patience (int, optional): Number of epochs to wait for improvement before early stopping. Default is None.
        verbose (bool, optional): If True, displays messages for each validation loss improvement. Default is False.
        model_title (str, optional): Title for the model, used in plot titles. Default is "Model".
        save (bool, optional): If True, saves the model's performance plot. Default is False.
    """
    
    print(f"Training of {model_title} starts!")
    start_time = time.time()
    
    # Lists to store loss values for plotting later
    train_loss_history, val_loss_history = [], []
    epochs_completed = 0
    
    # Set up optimizer and early stopping
    optimizer = Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience, verbose=verbose)

    # Training loop across specified epochs
    for epoch in range(epochs):
        epochs_completed += 1
        model.train()
        train_loss, val_loss = 0.0, 0.0  # Track epoch loss
        
        # Progress bar for the training loop
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training", leave=False) as train_pbar:
            for train_batch in train_pbar:
                # Move data to device and forward pass
                inputs, labels = train_batch[0].to(device), train_batch[1].to(device)
                predictions = model(inputs)
                loss = loss_fn(predictions, labels)
                
                # Apply regularization if specified
                if regularization:
                    reg_term = sum((p.pow(2.0) if regularization == 'L2' else p.abs()).sum() for p in model.parameters())
                    loss += reg_lambda * reg_term
                
                # Backpropagation and optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update training loss and progress bar
                train_loss += loss.item()
                train_pbar.set_postfix({"Batch Loss": loss.item()})
        
        # Validation loop, evaluating on the validation dataset
        model.eval()
        with torch.no_grad():
            with tqdm(val_loader, desc="Validation", leave=False) as val_pbar:
                for val_batch in val_pbar:
                    inputs, labels = val_batch[0].to(device), val_batch[1].to(device)
                    predictions = model(inputs)
                    batch_loss = loss_fn(predictions, labels).item()
                    val_loss += batch_loss
                    val_pbar.set_postfix({"Batch Loss": batch_loss})

        # Average losses over all batches in the epoch
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        print(f"Epoch: {epoch + 1}/{epochs} | Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        
        # Check for early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Plot training and validation loss history
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(range(1, epochs_completed + 1), train_loss_history, label='Training Loss', color="red", linewidth=2.5)
    ax.plot(range(1, epochs_completed + 1), val_loss_history, label='Validation Loss', color="blue", linewidth=2.5)
    ax.set_title(model_title, fontsize=15)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_xlabel("Epochs", fontsize=13)
    plt.legend()
    if save:
        plt.savefig(f"{model_title}.png")
    plt.show()

    # Load the best model checkpoint saved by early stopping
    model.load_state_dict(torch.load("logs/smp/checkpoint.pt"))
    
    # Calculate total training time
    elapsed_mins, elapsed_secs = divmod(time.time() - start_time, 60)
    print(f"\nTraining Completed in {int(elapsed_mins):02d}m {elapsed_secs:.2f}s.")



def test_scores(model, test_loader):
    # StatScores for TP, FP, TN, FN and support
    stat_scores = torchmetrics.StatScores(num_classes=5, task="multiclass", average='none').to(device)
    acc = torchmetrics.Accuracy(num_classes=5, average="micro", task="multiclass").to(device)
    jaccard = torchmetrics.JaccardIndex(num_classes=5, task="multiclass").to(device)
    
    model.eval()
    
    # Dictionaries to accumulate class probabilities and sample counts
    class_probs = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    num_samples = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for X, y in test_loader:
        X = X.to(device)
        y = y.to(device)

        with torch.no_grad():
            logits = F.softmax(model(X), dim=1)
            aggr = torch.max(logits, dim=1)
            preds = aggr[1]
            probs = aggr[0]
            
            # Accumulate probabilities and sample counts for each class
            for label in class_probs.keys():
                class_probs[label] += probs[preds == label].sum()
                num_samples[label] += (preds == label).sum().item()

            # Update metrics
            stat_scores.update(preds, y)
            acc.update(preds, y)
            jaccard.update(preds, y)

    # Final calculation for class probabilities, avoiding division by zero
    for label in class_probs.keys():
        if num_samples[label] > 0:
            class_probs[label] /= num_samples[label]
        else:
            class_probs[label] = 0  # or another suitable default value

    # Extract TP, FP, TN, FN, and support for each class from stat_scores
    stat_scores_results = stat_scores.compute()  # Shape: [num_classes, 5] -> [TP, FP, TN, FN, Support] for each class
    stat_scores_dict = {}
    for i in range(5):  # Assuming 5 classes
        stat_scores_dict[i] = {
            "true_positive": stat_scores_results[i, 0].item(),
            "false_positive": stat_scores_results[i, 1].item(),
            "true_negative": stat_scores_results[i, 2].item(),
            "false_negative": stat_scores_results[i, 3].item(),
            "support": stat_scores_results[i, 4].item()
        }

    return stat_scores_dict, acc.compute(), jaccard.compute(), class_probs


def evaluation_report(classes, scores, acc, jaccard, class_probs):
    print(f" {'Class':<20}{'Precision':<15}{'Recall':<15}{'F1-score':<15}{'Support'}\n")

    # Converting tensors to floats for printing
    acc = float(acc.cpu())
    jaccard = float(jaccard.cpu())

    # Iterate over each class and calculate metrics
    for i, target in enumerate(classes):
        true_positive = scores[i]["true_positive"]
        false_positive = scores[i]["false_positive"]
        false_negative = scores[i]["false_negative"]
        support = scores[i]["support"]

        # Precision, Recall, F1 calculations
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"* {target:<20}{precision:<15.2f}{recall:<15.2f}{f1:<15.2f}{support}")

    # Print overall accuracy and Jaccard score
    print(f"\n- Total accuracy: {acc:.4f}")
    print(f"- Mean IoU (Jaccard Index): {jaccard:.4f}\n")

    # Print class probabilities
    print("- Class probabilities")
    for idx, prob in class_probs.items():
        print(f"* {classes[idx]:<10}: {float(prob.cpu()):.3f}")

class EarlyStopping:
    """Early stopping utility to halt training when validation loss does not improve after a set patience period."""

    def __init__(self, patience=7, verbose=False, delta=0, checkpoint_path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): Number of epochs to wait for an improvement in validation loss.
                            Training will stop if no improvement is observed for this many epochs. Default is 7.
            verbose (bool): If True, logs a message each time validation loss improves. Default is False.
            delta (float): Minimum change in validation loss to qualify as an improvement. Default is 0.
            checkpoint_path (str): File path to save the best model checkpoint. Default is 'checkpoint.pt'.
            trace_func (function): Function for logging output, e.g., `print` for console logging. Default is print.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.trace_func = trace_func

        # Internal state variables
        self.counter = 0  # Counts the epochs with no improvement
        self.best_score = None  # Tracks the best validation loss score
        self.early_stop = False  # Flag to trigger early stopping
        self.min_val_loss = np.Inf  # Tracks the minimum validation loss encountered

    def __call__(self, val_loss, model):
        """Checks if validation loss has improved and decides whether to stop training early.
        
        Args:
            val_loss (float): Current epoch's validation loss.
            model (torch.nn.Module): Model to save if validation loss improves.
        """
        # Calculate score as the negative of validation loss (lower loss is better)
        score = -val_loss

        # Initialize best_score if this is the first epoch
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        
        # Check if validation loss has improved by at least `delta`
        elif score < self.best_score + self.delta:
            self.counter += 1  # Increment the counter if no improvement
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            
            # Stop training if counter reaches patience limit
            if self.counter >= self.patience:
                self.early_stop = True
        
        # Reset counter if there is an improvement in validation loss
        else:
            self.best_score = score  # Update best score
            self.save_checkpoint(val_loss, model)  # Save model checkpoint
            self.counter = 0  # Reset counter

    def save_checkpoint(self, val_loss, model):
        """Saves the model when validation loss decreases.
        
        Args:
            val_loss (float): Current epoch's validation loss.
            model (torch.nn.Module): Model to save if validation loss improves.
        """
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.min_val_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        # Save model state to the specified checkpoint path
        torch.save(model.state_dict(), self.checkpoint_path)
        self.min_val_loss = val_loss  # Update minimum validation loss
        
import matplotlib

labels_cmap = matplotlib.colors.ListedColormap(["#000000", "#A9A9A9", "#8B8680", "#D3D3D3", "#FFFFFF"])
def plot_predictions(model, train_set, title, num_samples = 4, seed = 42, w = 10, h = 10, save_title = None, indices = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(seed)
    if indices == None:
        indices = np.random.randint(low = 0, high = len(train_set), size = num_samples)
    fig, ax = plt.subplots(figsize = (w,h), nrows = num_samples, ncols = 3)
    model.eval()
    for i,idx in enumerate(indices):
        X,y = train_set.__getitem__(idx)
        X_dash = X[None,:,:,:].to(device)
        preds = torch.argmax(model(X_dash), dim = 1)
        preds = torch.squeeze(preds).detach().cpu().numpy()
        ax[i,0].imshow(np.transpose(X.cpu(), (2,1,0)))
        ax[i,0].set_title("True Image")
        ax[i,0].axis("off")
        ax[i,1].imshow(y, cmap = labels_cmap, interpolation = None, vmin = -0.5, vmax = 4.5)
        ax[i,1].set_title("Labels")
        ax[i,1].axis("off")
        ax[i,2].imshow(preds, cmap = labels_cmap, interpolation = None, vmin = -0.5, vmax = 4.5)
        ax[i,2].set_title("Predictions")
        ax[i,2].axis("off")
    fig.suptitle(title, fontsize = 20)
    plt.tight_layout()
    if save_title is not None:
        plt.savefig(save_title + ".png")
    plt.show()
