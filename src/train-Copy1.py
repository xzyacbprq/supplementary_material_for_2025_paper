import sys
sys.path.append('libs/smp')
sys.path.append('libs/mmsegmentation')
from mmseg.evaluation import IoUMetric
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
from torch.optim import Adam, SGD
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

from torch.utils.checkpoint import checkpoint
'''
def custom_forward(*inputs):
    return model(*inputs)

outputs = checkpoint(custom_forward, inputs)
'''
def train(model, 
          train_loader, 
          val_loader, 
          epochs, 
          optimizer, 
          scheduler, 
          evaluator = None, 
          clip_grad = None,
          regularization=None, 
          reg_lambda=None, 
          patience=None, 
          verbose=False, 
          device = 'cuda', 
          output_dir = 'logs/test'):
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
        save (bool, optional): If True, saves the model's performance plot. Default is False.
    """
    
    print(f"Training starts ...")
    start_time = time.time()
    
    train_loss_history, val_loss_history = [], []
    epochs_completed = 0
    
    # Set up optimizer and early stopping
    early_stopping = EarlyStopping(patience=patience, checkpoint_path = f'{output_dir}/checkpoint.pt', verbose=verbose)
    
    # Training loop across specified epochs
    for epoch in range(epochs):
        epochs_completed += 1
        model.train()
        train_loss, val_loss = 0.0, 0.0  # Track epoch loss
        
        # Progress bar for the training loop
        batch_loss_dict = {'loss' : 0.0}#, 'val_loss' : 0.0}
        n_smp = len(train_loader)
        for train_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training", leave=False):
                train_batch['inputs'] = train_batch['inputs'].float().to(device)
                for key in train_batch['data_samples'].keys():
                    train_batch['data_samples'][key] = train_batch['data_samples'][key].to(device, non_blocking = True)
                loss_dict   = checkpoint(model.loss, train_batch['inputs'], train_batch['data_samples'])
                
                loss = 0
                for key in loss_dict.keys():
                    if 'loss' in key:
                        loss += loss_dict[key] #torch.nan_to_num(loss_dict[key])
                    if key not in batch_loss_dict:
                        batch_loss_dict[key] = ((1/n_smp) * loss_dict[key]).detach().cpu().numpy()
                    else:
                        batch_loss_dict[key] += ((1/n_smp) * loss_dict[key]).detach().cpu().numpy()
                batch_loss_dict['loss'] += (1/n_smp) * loss  
                
                if regularization:
                    reg_term = sum((p.pow(2.0) if regularization == 'L2' else p.abs()).sum() for p in model.parameters())
                    loss += reg_lambda * reg_term
                
                # Backpropagation and optimizer step
                optimizer.zero_grad()
                loss.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad, norm_type=2.0, error_if_nonfinite=False, foreach=None)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                
                # Update training loss and progress bar
                train_loss += loss.item()
            
        # Validation loop, evaluating on the validation dataset
        model.eval()
        evaluator.results = []
        n_smp = len(val_loader)
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="Validation", leave=False):
                val_batch['inputs'] = val_batch['inputs'].float().to(device)
                for key in val_batch['data_samples'].keys():
                    val_batch['data_samples'][key] = val_batch['data_samples'][key].to(device, non_blocking = True)
                preds   = model._forward(val_batch['inputs'], val_batch['data_samples'])
                evaluator.process(data_samples = preds, data_batch = None)
                    
        res = evaluator.compute_metrics(evaluator.results)
        batch_loss_dict.update(res)
        val_loss = batch_loss_dict['mIoU1']
        path = f'{output_dir}/losses.csv'
        keys = list(batch_loss_dict.keys())
        vals = list(map(str,[batch_loss_dict[key] for key in keys]))
        if not os.path.exists(path):
            os.makedirs(output_dir,exist_ok = True)
            with open(path, 'w') as file:
                file.write(','.join(keys)+'\n')
        with open(path, 'a') as file:
            file.write(','.join(vals)+'\n')
        
        # Average losses over all batches in the epoch
        train_loss_history.append(batch_loss_dict['loss'].detach().cpu().numpy())
        val_loss_history.append(val_loss)
        
        print(f"Epoch: {epoch + 1}/{epochs} | Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Plot training and validation loss history
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(range(1, epochs_completed + 1), train_loss_history, label='Training Loss', color="red", linewidth=2.5)
    ax.plot(range(1, epochs_completed + 1), val_loss_history, label='Validation Loss', color="blue", linewidth=2.5)
    ax.set_title('Loss vs epoch', fontsize=15)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_xlabel("Epochs", fontsize=13)
    plt.legend()
    plt.savefig(f"{output_dir}/loss_viz.png")
    plt.show()

    # Load the best model checkpoint saved by early stopping
    model.load_state_dict(torch.load(f"{output_dir}/checkpoint.pt"))
    
    # Calculate total training time
    elapsed_mins, elapsed_secs = divmod(time.time() - start_time, 60)
    print(f"\nTraining Completed in {int(elapsed_mins):02d}m {elapsed_secs:.2f}s.")






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
        self.min_val_score = 0.0  # Tracks the minimum validation loss encountered

    def __call__(self, score, model):
        """Checks if validation loss has improved and decides whether to stop training early.
        
        Args:
            val_loss (float): Current epoch's validation loss.
            model (torch.nn.Module): Model to save if validation loss improves.
        """

        # Initialize best_score if this is the first epoch
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        
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
            self.save_checkpoint(score, model)  # Save model checkpoint
            self.counter = 0  # Reset counter

    def save_checkpoint(self, score, model):
        """Saves the model when validation loss decreases.
        
        Args:
            val_loss (float): Current epoch's validation loss.
            model (torch.nn.Module): Model to save if validation loss improves.
        """
        if self.verbose:
            self.trace_func(f'Validation IoU improved ({self.min_val_score:.6f} --> {score:.6f}).  Saving model ...')
        
        # Save model state to the specified checkpoint path
        torch.save(model.state_dict(), self.checkpoint_path)
        self.min_val_score = score  # Update minimum validation loss

def test_scores(model, test_loader, device = 'cuda'):
    # StatScores for TP, FP, TN, FN and support
    stat_scores = torchmetrics.StatScores(num_classes=5, task="multiclass", average='none').to(device)
    acc = torchmetrics.Accuracy(num_classes=5, average="micro", task="multiclass").to(device)
    jaccard = torchmetrics.JaccardIndex(num_classes=5, task="multiclass").to(device)
    
    model.eval()
    
    # Dictionaries to accumulate class probabilities and sample counts
    class_probs = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    num_samples = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for sample in tqdm(test_loader):
        X = sample['inputs'].float().to(device)
        y = sample['data_samples']['gt_sem_seg'].to(device)

        with torch.no_grad():
            preds = model._forward(X, {})['pred_sem_seg'].squeeze()
            # Accumulate probabilities and sample counts for each class
            #for label in class_probs.keys():
            #class_probs[label] += probs[preds == label].sum()
            #num_samples[label] += (preds == label).sum().item()
            
            # Update metrics
            stat_scores.update(preds, y)
            acc.update(preds, y)
            jaccard.update(preds, y)

    # Final calculation for class probabilities, avoiding division by zero
    '''
    for label in class_probs.keys():
        if num_samples[label] > 0:
            class_probs[label] /= num_samples[label]
        else:
            class_probs[label] = 0  # or another suitable default value
    '''
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
    return stat_scores_dict, acc.compute(), jaccard.compute()
#test_scores(model, ValDataloader, device = 'cuda')

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
