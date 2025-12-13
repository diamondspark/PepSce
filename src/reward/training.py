import sys
sys.path.append('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/PepSce_Final/src')
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
import math
from pathlib import Path
from architecture import PeptideRegressionModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from sklearn.model_selection import KFold, train_test_split
import json
from datetime import datetime
from torch.utils.data import DataLoader
from peptide_data import PeptideDataset, process_peptide_csv
import wandb
wandb.login()
import yaml
import matplotlib.pyplot as plt
import seaborn as sns


def plot_predictions(y_true, y_pred, pearson, fold_num, save_dir, dataset_name='validation'):
    """
    Create and save scatter plot of predictions vs true values
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    pearson : float
        Pearson correlation coefficient
    fold_num : int
        Fold number
    save_dir : Path
        Directory to save plots
    dataset_name : str
        Name of dataset (e.g., 'validation', 'test')
    """
    # Flatten arrays
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 10))
    
    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    # Add diagonal line (perfect predictions)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Add best fit line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_true, p(y_true), "b-", alpha=0.8, lw=2, label=f'Best Fit: y={z[0]:.3f}x+{z[1]:.3f}')
    
    # Labels and title
    plt.xlabel('True Values', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Values', fontsize=14, fontweight='bold')
    plt.title(f'Fold {fold_num} - {dataset_name.capitalize()} Set\nPearson r = {pearson:.4f}', 
              fontsize=16, fontweight='bold')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'N = {len(y_true)}\n'
    stats_text += f'Pearson r = {pearson:.4f}\n'
    stats_text += f'MSE = {np.mean((y_true - y_pred)**2):.4f}\n'
    stats_text += f'MAE = {np.mean(np.abs(y_true - y_pred)):.4f}'
    
    plt.text(0.05, 0.95, stats_text, 
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=11)
    
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    
    # Save figure
    save_path = save_dir / f'fold_{fold_num}_{dataset_name}_predictions.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved plot: {save_path}")
    
    # Log to wandb if available
    try:
        wandb.log({f"fold_{fold_num}/{dataset_name}_scatter": wandb.Image(str(save_path))})
    except:
        pass


def plot_all_folds_summary(fold_results, save_dir):
    """
    Create summary plot comparing all folds
    
    Parameters
    ----------
    fold_results : list
        List of fold result dictionaries
    save_dir : Path
        Directory to save plots
    """
    n_folds = len(fold_results)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Validation Loss
    fold_nums = [r['fold_num'] for r in fold_results]
    val_losses = [r['best_val_loss'] for r in fold_results]
    
    axes[0].bar(fold_nums, val_losses, color='skyblue', edgecolor='black', linewidth=1.5)
    axes[0].axhline(y=np.mean(val_losses), color='r', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(val_losses):.4f}')
    axes[0].set_xlabel('Fold Number', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Validation Loss by Fold', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Pearson Correlation
    pearsons = [r['best_pearson'] for r in fold_results]
    
    axes[1].bar(fold_nums, pearsons, color='lightcoral', edgecolor='black', linewidth=1.5)
    axes[1].axhline(y=np.mean(pearsons), color='r', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(pearsons):.4f}')
    axes[1].set_xlabel('Fold Number', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Pearson Correlation', fontsize=12, fontweight='bold')
    axes[1].set_title('Pearson Correlation by Fold', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save figure
    save_path = save_dir / 'cv_folds_summary.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved CV summary plot: {save_path}")
    
    # Log to wandb if available
    try:
        wandb.log({"cv_summary_plot": wandb.Image(str(save_path))})
    except:
        pass


def train_one_epoch(train_loader, model, optimizer, loss_fn, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    step = 0
    
    for inputs, target in train_loader:
        X_sequence, X_fixed = inputs
        
        # Move to device
        X_sequence = X_sequence.float().to(device)
        X_fixed = X_fixed.float().to(device)
        target = target.float().to(device).reshape(-1, 1)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(X_sequence, X_fixed)
        loss = loss_fn(logits, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        step += 1
        
        # Clear cache periodically
        if step % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return total_loss / step


def validate_one_epoch(val_loader, model, loss_fn, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    step = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, target in val_loader:
            X_sequence, X_fixed = inputs
            
            # Move to device
            X_sequence = X_sequence.float().to(device)
            X_fixed = X_fixed.float().to(device)
            target = target.float().to(device).reshape(-1, 1)
            
            # Forward pass
            pred = model(X_sequence, X_fixed)
            loss = loss_fn(pred, target)
            
            total_loss += loss.item()
            step += 1
            
            # Store predictions
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    avg_loss = total_loss / step
    
    # Calculate Pearson correlation
    flat_preds = np.squeeze(np.array(all_preds))
    flat_labels = np.squeeze(np.array(all_labels))
    
    if len(flat_preds) > 1:
        pearson = pearsonr(flat_labels, flat_preds)[0]
    else:
        pearson = 0.0
    
    return avg_loss, pearson, all_preds, all_labels


def train_single_fold(fold_num, train_loader, val_loader, config, device, 
                     sample_sequence_shape, sample_fixed_shape, save_dir, run_name=None):
    """
    Train a single fold
    
    Returns
    -------
    dict
        Results including best model state, validation metrics, etc.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING FOLD {fold_num}")
    print(f"{'='*60}")
    
    # Build model
    model = build_model_from_config(config, sample_sequence_shape, sample_fixed_shape)
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['regression']['learning_rate'],
        weight_decay=config['regression'].get('weight_decay', 1e-6)
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config['regression'].get('scheduler_patience', 5),
        factor=config['regression'].get('scheduler_factor', 0.5),
        min_lr=config['regression'].get('scheduler_min_lr', 1e-7)
    )
    
    # Loss function
    loss_fn_name = config['regression'].get('loss_function', 'MSE')
    if loss_fn_name == 'MSE':
        loss_fn = nn.MSELoss()
    elif loss_fn_name == 'MAE':
        loss_fn = nn.L1Loss()
    elif loss_fn_name == 'Huber':
        loss_fn = nn.HuberLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_fn_name}")
    
    # Training loop
    best_loss = math.inf
    best_pearson = -1
    best_model_state = None
    best_preds = None
    best_labels = None
    early_stopping_counter = 0
    
    epochs = config['regression']['epochs']
    early_stopping_patience = config['regression']['early_stopping_patience']
    
    for epoch in range(epochs):
        # Training
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, device)
        
        # Validation
        val_loss, pearson, all_preds, all_labels = validate_one_epoch(
            val_loader, model, loss_fn, device
        )
        
        # Log to wandb if enabled
        if config.get('wandb', True) and run_name:
            wandb.log({
                f"fold_{fold_num}/train_loss": train_loss,
                f"fold_{fold_num}/val_loss": val_loss,
                f"fold_{fold_num}/pearson": pearson,
                f"fold_{fold_num}/learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch
            })
        
        if epoch % 10 == 0:
            print(f"Fold {fold_num} | Epoch {epoch:4d} | Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | Pearson: {pearson:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_pearson = pearson
            best_model_state = model.state_dict().copy()
            best_preds = all_preds
            best_labels = all_labels
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"\nFold {fold_num} Complete:")
    print(f"  Best Val Loss: {best_loss:.6f}")
    print(f"  Best Pearson: {best_pearson:.4f}")
    
    # Plot predictions for this fold
    print(f"\nGenerating plots for Fold {fold_num}...")
    plot_predictions(
        y_true=np.array(best_labels),
        y_pred=np.array(best_preds),
        pearson=best_pearson,
        fold_num=fold_num,
        save_dir=save_dir,
        dataset_name='validation'
    )
    
    return {
        'fold_num': fold_num,
        'best_val_loss': best_loss,
        'best_pearson': best_pearson,
        'best_model_state': best_model_state,
        'best_preds': best_preds,
        'best_labels': best_labels,
        'final_epoch': epoch
    }


def k_fold_cross_validation(df_train, config, device, n_folds=3):
    """
    Perform K-fold cross-validation
    
    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataframe with descriptors
    config : dict
        Configuration dictionary
    device : torch.device
        Device to train on
    n_folds : int
        Number of folds for cross-validation
    
    Returns
    -------
    dict
        Results from all folds including best model
    """
    print(f"\n{'='*60}")
    print(f"STARTING {n_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*60}")
    
    # Create plots directory
    save_dir = Path(config['regression']['save_dir'])
    plots_dir = save_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract features
    X_sequence = df_train['descriptors_ifeat'].tolist()
    X_fixed = np.vstack(df_train['descriptors'].tolist())
    y = np.array(df_train['Normalized_Potency'])
    
    # Get sample shapes
    sample_sequence_shape = np.array(X_sequence[0]).shape
    sample_fixed_shape = X_fixed[0].shape
    
    print(f"\nData Statistics:")
    print(f"  Total samples: {len(y)}")
    print(f"  Sequence shape (example): {sample_sequence_shape}")
    print(f"  Fixed shape: {sample_fixed_shape}")
    print(f"  Target range: [{y.min():.4f}, {y.max():.4f}]")
    
    # K-Fold split
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=config['regression']['random_seed'])
    
    # Normalize fixed features (fit on all training data)
    scaler_type = config['regression'].get('scaler_type', 'StandardScaler')
    if scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler().fit(X_fixed)
    else:
        scaler = StandardScaler().fit(X_fixed)
    
    X_fixed_norm = scaler.transform(X_fixed)
    
    # Store results from each fold
    fold_results = []
    
    # Initialize wandb for cross-validation
    if config.get('wandb', True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"cv_{n_folds}fold_{timestamp}"
        wandb.init(
            project=config['regression']['wandb_project'],
            name=run_name,
            config=config['regression']
        )
    else:
        run_name = None
    
    # Train each fold
    for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_sequence), 1):
        # Split data
        X_seq_train = [X_sequence[i] for i in train_idx]
        X_seq_val = [X_sequence[i] for i in val_idx]
        X_fix_train = X_fixed_norm[train_idx]
        X_fix_val = X_fixed_norm[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]
        
        # Create datasets
        train_dataset = PeptideDataset(X_seq_train, X_fix_train, y_train)
        val_dataset = PeptideDataset(X_seq_val, X_fix_val, y_val)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['regression']['batch_size'],
            shuffle=True,
            num_workers=config['regression']['num_workers'],
            pin_memory=config['regression'].get('pin_memory', True)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['regression']['batch_size'],
            shuffle=False,
            num_workers=config['regression']['num_workers'],
            pin_memory=config['regression'].get('pin_memory', True)
        )
        
        # Train fold
        fold_result = train_single_fold(
            fold_num, train_loader, val_loader, config, device,
            sample_sequence_shape, sample_fixed_shape, plots_dir, run_name
        )
        fold_results.append(fold_result)
    
    # Create summary plot for all folds
    print(f"\n{'='*60}")
    print("CREATING CROSS-VALIDATION SUMMARY PLOTS")
    print(f"{'='*60}")
    plot_all_folds_summary(fold_results, plots_dir)
    
    # Calculate average performance
    avg_val_loss = np.mean([r['best_val_loss'] for r in fold_results])
    avg_pearson = np.mean([r['best_pearson'] for r in fold_results])
    
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Average Validation Loss: {avg_val_loss:.6f} ± {np.std([r['best_val_loss'] for r in fold_results]):.6f}")
    print(f"Average Pearson: {avg_pearson:.4f} ± {np.std([r['best_pearson'] for r in fold_results]):.4f}")
    
    # Log to wandb
    if config.get('wandb', True):
        wandb.log({
            "cv/avg_val_loss": avg_val_loss,
            "cv/avg_pearson": avg_pearson,
            "cv/std_val_loss": np.std([r['best_val_loss'] for r in fold_results]),
            "cv/std_pearson": np.std([r['best_pearson'] for r in fold_results])
        })
        wandb.finish()
    
    # Select best fold (lowest validation loss)
    best_fold = min(fold_results, key=lambda x: x['best_val_loss'])
    print(f"\nBest Fold: {best_fold['fold_num']}")
    print(f"  Val Loss: {best_fold['best_val_loss']:.6f}")
    print(f"  Pearson: {best_fold['best_pearson']:.4f}")
    
    return {
        'fold_results': fold_results,
        'best_fold': best_fold,
        'avg_val_loss': avg_val_loss,
        'avg_pearson': avg_pearson,
        'scaler': scaler,
        'sample_sequence_shape': sample_sequence_shape,
        'sample_fixed_shape': sample_fixed_shape,
        'plots_dir': plots_dir
    }


def build_model_from_config(config, sample_sequence_shape=None, sample_fixed_shape=None):
    """
    Build model from configuration
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    sample_sequence_shape : tuple, optional
        Example shape of sequence descriptors (channels, seq_len)
    sample_fixed_shape : tuple, optional
        Example shape of fixed descriptors (feature_dim,)
    
    Returns
    -------
    model : PeptideRegressionModel
    """
    reg_config = config['regression']
    
    # Auto-detect dimensions if not specified
    cnn_in_channels = reg_config.get('cnn_in_channels')
    mlp_input_dim = reg_config.get('mlp_input_dim')
    
    if cnn_in_channels is None and sample_sequence_shape is not None:
        cnn_in_channels = sample_sequence_shape[0]
        print(f"Auto-detected CNN input channels: {cnn_in_channels}")
    elif cnn_in_channels is None:
        cnn_in_channels = 14
    
    if mlp_input_dim is None and sample_fixed_shape is not None:
        mlp_input_dim = sample_fixed_shape[0] if len(sample_fixed_shape) == 1 else sample_fixed_shape[-1]
        print(f"Auto-detected MLP input dimension: {mlp_input_dim}")
    elif mlp_input_dim is None:
        mlp_input_dim = 139
    
    model = PeptideRegressionModel()
    #     cnn_in_channels=cnn_in_channels,
    #     cnn_out_channels=reg_config.get('cnn_out_channels', 5),
    #     cnn_kernel_sizes=reg_config.get('cnn_kernel_sizes', [2, 3, 4, 5]),
    #     cnn_fc_dims=reg_config.get('cnn_fc_dims', [735, 128, 64]),
    #     mlp_input_dim=mlp_input_dim,
    #     mlp_hidden_dims=reg_config.get('mlp_hidden_dims', [100, 64]),
    #     mlp_use_batchnorm=reg_config.get('mlp_use_batchnorm', True),
    #     fusion_hidden_dims=reg_config.get('fusion_hidden_dims', [64, 32]),
    #     dropout=reg_config.get('dropout', 0.0)
    # )
    
    return model


def main(config_path='config.yml', csv_path=None):
    """
    Main training function with 3-fold cross-validation
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
    csv_path : str
        Path to CSV file with columns: Peptide Name, Sequence, Potency
    """
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config.get('device', 'cuda') == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Get CSV path from config if not provided
    if csv_path is None:
        csv_path = config['data']['csv_path']
    
    print(f"\n{'='*60}")
    print(f"DATA PREPROCESSING")
    print(f"{'='*60}")
    
    df_all, normalization_params = process_peptide_csv(
        csv_path, 
        normalize_target=config['regression'].get('normalize_target', True)
    )
    
    # Split data: train (80%) and test (20%)
    test_size = config['regression'].get('test_size', 0.2)
    random_seed = config['regression'].get('random_seed', 42)
    
    df_train, df_test = train_test_split(
        df_all, 
        test_size=test_size, 
        random_state=random_seed,
        shuffle=True
    )
    
    print(f"\nData Split:")
    print(f"  Training set: {len(df_train)} samples ({(1-test_size)*100:.0f}%)")
    print(f"  Test set: {len(df_test)} samples ({test_size*100:.0f}%)")
    
    # Perform K-fold cross-validation on training data
    n_folds = config['regression'].get('n_folds', 3)
    cv_results = k_fold_cross_validation(df_train, config, device, n_folds=n_folds)
    
    # Save results
    save_dir = Path(config['regression']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = cv_results['plots_dir']
    
    # Save scaler
    scaler_path = save_dir / 'fixed_features_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(cv_results['scaler'], f)
    print(f"\n✓ Saved scaler: {scaler_path}")
    
    # Save normalization parameters
    if normalization_params:
        norm_path = save_dir / 'normalization_params.json'
        with open(norm_path, 'w') as f:
            json.dump(normalization_params, f, indent=2)
        print(f"✓ Saved normalization params: {norm_path}")
    
    # Build final model with best fold weights
    final_model = build_model_from_config(
        config, 
        cv_results['sample_sequence_shape'],
        cv_results['sample_fixed_shape']
    )
    final_model.load_state_dict(cv_results['best_fold']['best_model_state'])
    final_model = final_model.to(device)
    
    # Evaluate on test set
    print(f"\n{'='*60}")
    print(f"EVALUATING ON TEST SET")
    print(f"{'='*60}")
    
    X_seq_test = df_test['descriptors_ifeat'].tolist()
    X_fix_test = np.vstack(df_test['descriptors'].tolist())
    y_test = np.array(df_test['Normalized_Potency'])
    
    # Normalize fixed features
    X_fix_test_norm = cv_results['scaler'].transform(X_fix_test)
    
    # Create test dataset
    test_dataset = PeptideDataset(X_seq_test, X_fix_test_norm, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['regression']['batch_size'],
        shuffle=False,
        num_workers=config['regression']['num_workers'],
        pin_memory=config['regression'].get('pin_memory', True)
    )
    
    # Evaluate
    loss_fn = nn.MSELoss()
    test_loss, test_pearson, test_preds, test_labels = validate_one_epoch(
        test_loader, final_model, loss_fn, device
    )
    
    print(f"\nTest Set Performance:")
    print(f"  Test Loss: {test_loss:.6f}")
    print(f"  Test Pearson: {test_pearson:.4f}")
    
    # Plot test set predictions
    print(f"\nGenerating test set plot...")
    plot_predictions(
        y_true=np.array(test_labels),
        y_pred=np.array(test_preds),
        pearson=test_pearson,
        fold_num='FINAL',
        save_dir=plots_dir,
        dataset_name='test'
    )
    
    # Save best model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"best_model_{n_folds}fold_test_loss_{test_loss:.6f}_pearson_{test_pearson:.4f}_{timestamp}.pt"
    model_path = save_dir / model_filename
    
    checkpoint = {
        'model_state_dict': final_model.state_dict(),
        'config': config,
        'cv_results': {
            'avg_val_loss': cv_results['avg_val_loss'],
            'avg_pearson': cv_results['avg_pearson'],
            'best_fold_num': cv_results['best_fold']['fold_num'],
            'best_fold_val_loss': cv_results['best_fold']['best_val_loss'],
            'best_fold_pearson': cv_results['best_fold']['best_pearson'],
        },
        'test_results': {
            'test_loss': test_loss,
            'test_pearson': test_pearson,
        },
        'normalization_params': normalization_params,
        'timestamp': timestamp
    }
    
    torch.save(checkpoint, model_path)
    print(f"\n✓ Saved best model: {model_path}")
    
    # Save cross-validation summary
    cv_summary = {
        'n_folds': n_folds,
        'fold_results': [
            {
                'fold_num': r['fold_num'],
                'best_val_loss': float(r['best_val_loss']),
                'best_pearson': float(r['best_pearson']),
                'final_epoch': r['final_epoch']
            }
            for r in cv_results['fold_results']
        ],
        'avg_val_loss': float(cv_results['avg_val_loss']),
        'avg_pearson': float(cv_results['avg_pearson']),
        'test_loss': float(test_loss),
        'test_pearson': float(test_pearson),
        'best_fold_num': cv_results['best_fold']['fold_num'],
        'training_date': timestamp
    }
    
    summary_path = save_dir / f'cv_summary_{timestamp}.json'
    with open(summary_path, 'w') as f:
        json.dump(cv_summary, f, indent=2)
    print(f"✓ Saved CV summary: {summary_path}")
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best model saved to: {model_path}")
    print(f"CV Average Val Loss: {cv_results['avg_val_loss']:.6f}")
    print(f"CV Average Pearson: {cv_results['avg_pearson']:.4f}")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test Pearson: {test_pearson:.4f}")
    print(f"\nAll plots saved to: {plots_dir}")
    print(f"  - Fold validation plots: fold_X_validation_predictions.png")
    print(f"  - Test set plot: fold_FINAL_test_predictions.png")
    print(f"  - CV summary plot: cv_folds_summary.png")

    return final_model, cv_results, {'test_loss': test_loss, 'test_pearson': test_pearson}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train Peptide Regression Model with 3-Fold CV')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to peptide CSV file')
    args = parser.parse_args()
    
    main(args.config, csv_path=args.data)