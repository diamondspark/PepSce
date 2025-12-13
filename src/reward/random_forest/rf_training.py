"""
Random Forest Regression with 3-Fold Cross-Validation for Peptide Potency Prediction

This script trains a Random Forest model on peptide descriptors with proper
cross-validation and evaluation.
"""

import sys
sys.path.append('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/Generative_Modeling/ACP/PepSce_Final/src')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from reward.peptide_data import process_peptide_csv


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
    fold_num : int or str
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
    plt.title(f'Random Forest - Fold {fold_num} - {dataset_name.capitalize()} Set\nPearson r = {pearson:.4f}', 
              fontsize=16, fontweight='bold')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    mse = np.mean((y_true - y_pred)**2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    
    stats_text = f'N = {len(y_true)}\n'
    stats_text += f'Pearson r = {pearson:.4f}\n'
    stats_text += f'R² = {r2:.4f}\n'
    stats_text += f'RMSE = {rmse:.4f}\n'
    stats_text += f'MAE = {mae:.4f}'
    
    plt.text(0.05, 0.95, stats_text, 
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=11)
    
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    
    # Save figure
    save_path = save_dir / f'rf_fold_{fold_num}_{dataset_name}_predictions.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved plot: {save_path}")


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
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    fold_nums = [r['fold_num'] for r in fold_results]
    
    # Plot 1: MSE
    mses = [r['mse'] for r in fold_results]
    axes[0, 0].bar(fold_nums, mses, color='skyblue', edgecolor='black', linewidth=1.5)
    axes[0, 0].axhline(y=np.mean(mses), color='r', linestyle='--', linewidth=2, 
                        label=f'Mean: {np.mean(mses):.4f}')
    axes[0, 0].set_xlabel('Fold Number', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('MSE', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Mean Squared Error by Fold', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Pearson Correlation
    pearsons = [r['pearson'] for r in fold_results]
    axes[0, 1].bar(fold_nums, pearsons, color='lightcoral', edgecolor='black', linewidth=1.5)
    axes[0, 1].axhline(y=np.mean(pearsons), color='r', linestyle='--', linewidth=2,
                        label=f'Mean: {np.mean(pearsons):.4f}')
    axes[0, 1].set_xlabel('Fold Number', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Pearson Correlation', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Pearson Correlation by Fold', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([0, 1])
    
    # Plot 3: R²
    r2s = [r['r2'] for r in fold_results]
    axes[1, 0].bar(fold_nums, r2s, color='lightgreen', edgecolor='black', linewidth=1.5)
    axes[1, 0].axhline(y=np.mean(r2s), color='r', linestyle='--', linewidth=2,
                        label=f'Mean: {np.mean(r2s):.4f}')
    axes[1, 0].set_xlabel('Fold Number', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('R² Score', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('R² Score by Fold', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: MAE
    maes = [r['mae'] for r in fold_results]
    axes[1, 1].bar(fold_nums, maes, color='plum', edgecolor='black', linewidth=1.5)
    axes[1, 1].axhline(y=np.mean(maes), color='r', linestyle='--', linewidth=2,
                        label=f'Mean: {np.mean(maes):.4f}')
    axes[1, 1].set_xlabel('Fold Number', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('MAE', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Mean Absolute Error by Fold', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    save_path = save_dir / 'rf_cv_folds_summary.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved CV summary plot: {save_path}")


def plot_feature_importance(model, feature_names, save_dir, top_n=20):
    """
    Plot feature importance from Random Forest
    
    Parameters
    ----------
    model : RandomForestRegressor
        Trained Random Forest model
    feature_names : list
        Names of features
    save_dir : Path
        Directory to save plots
    top_n : int
        Number of top features to display
    """
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot
    plt.barh(range(top_n), importances[indices], color='steelblue', edgecolor='black')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    # Save figure
    save_path = save_dir / 'rf_feature_importance.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved feature importance plot: {save_path}")


def flatten_features(X_sequence, X_fixed):
    """
    Flatten sequence and fixed features into a single feature vector
    
    Parameters
    ----------
    X_sequence : list
        List of sequence descriptors with shape (channels, seq_len)
    X_fixed : np.ndarray
        Fixed-size descriptors with shape (n_samples, n_features)
    
    Returns
    -------
    X_flat : np.ndarray
        Flattened features with shape (n_samples, total_features)
    feature_names : list
        Names of all features
    """
    n_samples = len(X_sequence)
    
    # Flatten sequence features
    seq_flat_list = []
    for seq in X_sequence:
        seq_flat_list.append(seq.flatten())
    
    X_seq_flat = np.array(seq_flat_list)
    
    # Combine with fixed features
    X_flat = np.concatenate([X_seq_flat, X_fixed], axis=1)
    
    # Create feature names
    seq_shape = X_sequence[0].shape
    feature_names = []
    
    # Sequence feature names
    for ch in range(seq_shape[0]):
        for pos in range(seq_shape[1]):
            feature_names.append(f'seq_ch{ch}_pos{pos}')
    
    # Fixed feature names
    for i in range(X_fixed.shape[1]):
        feature_names.append(f'fixed_{i}')
    
    return X_fixed, feature_names #X_flat, feature_names


def train_single_fold(fold_num, X_train, y_train, X_val, y_val, config, save_dir):
    """
    Train a single fold
    
    Parameters
    ----------
    fold_num : int
        Fold number
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets
    X_val : np.ndarray
        Validation features
    y_val : np.ndarray
        Validation targets
    config : dict
        Configuration dictionary
    save_dir : Path
        Directory to save plots
    
    Returns
    -------
    dict
        Results including model, predictions, and metrics
    """
    print(f"\n{'='*60}")
    print(f"TRAINING FOLD {fold_num}")
    print(f"{'='*60}")
    
    # Get Random Forest parameters from config
    rf_params = config['random_forest']
    
    # Initialize Random Forest
    model = RandomForestRegressor(
        n_estimators=rf_params.get('n_estimators', 100),
        max_depth=rf_params.get('max_depth', None),
        min_samples_split=rf_params.get('min_samples_split', 2),
        min_samples_leaf=rf_params.get('min_samples_leaf', 1),
        max_features=rf_params.get('max_features', 'sqrt'),
        random_state=config['random_forest']['random_seed'],
        n_jobs=rf_params.get('n_jobs', -1),
        verbose=0
    )
    
    print(f"Training Random Forest with {rf_params.get('n_estimators', 100)} trees...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predict on validation set
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    mse = np.mean((y_val - y_pred)**2)
    mae = np.mean(np.abs(y_val - y_pred))
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((y_val - y_pred)**2) / np.sum((y_val - np.mean(y_val))**2))
    pearson = pearsonr(y_val, y_pred)[0]
    
    print(f"\nFold {fold_num} Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.6f}")
    print(f"  Pearson: {pearson:.4f}")
    
    # Plot predictions
    print(f"\nGenerating plots for Fold {fold_num}...")
    plot_predictions(
        y_true=y_val,
        y_pred=y_pred,
        pearson=pearson,
        fold_num=fold_num,
        save_dir=save_dir,
        dataset_name='validation'
    )
    
    return {
        'fold_num': fold_num,
        'model': model,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson': pearson,
        'predictions': y_pred,
        'true_values': y_val
    }


def k_fold_cross_validation(X, y, feature_names, config, n_folds=3):
    """
    Perform K-fold cross-validation
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    feature_names : list
        Names of features
    config : dict
        Configuration dictionary
    n_folds : int
        Number of folds
    
    Returns
    -------
    dict
        Results from all folds including best model
    """
    print(f"\n{'='*60}")
    print(f"STARTING {n_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*60}")
    
    # Create plots directory
    save_dir = Path(config['random_forest']['save_dir'])
    plots_dir = save_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nData Statistics:")
    print(f"  Total samples: {len(y)}")
    print(f"  Total features: {X.shape[1]}")
    print(f"  Target range: [{y.min():.4f}, {y.max():.4f}]")
    
    # K-Fold split
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=config['random_forest']['random_seed'])
    
    # Store results from each fold
    fold_results = []
    
    # Train each fold
    for fold_num, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train fold
        fold_result = train_single_fold(
            fold_num, X_train, y_train, X_val, y_val, config, plots_dir
        )
        fold_results.append(fold_result)
    
    # Create summary plot for all folds
    print(f"\n{'='*60}")
    print("CREATING CROSS-VALIDATION SUMMARY PLOTS")
    print(f"{'='*60}")
    plot_all_folds_summary(fold_results, plots_dir)
    
    # Calculate average performance
    avg_mse = np.mean([r['mse'] for r in fold_results])
    avg_mae = np.mean([r['mae'] for r in fold_results])
    avg_r2 = np.mean([r['r2'] for r in fold_results])
    avg_pearson = np.mean([r['pearson'] for r in fold_results])
    
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Average MSE: {avg_mse:.6f} ± {np.std([r['mse'] for r in fold_results]):.6f}")
    print(f"Average MAE: {avg_mae:.6f} ± {np.std([r['mae'] for r in fold_results]):.6f}")
    print(f"Average R²: {avg_r2:.6f} ± {np.std([r['r2'] for r in fold_results]):.6f}")
    print(f"Average Pearson: {avg_pearson:.4f} ± {np.std([r['pearson'] for r in fold_results]):.4f}")
    
    # Select best fold (lowest MSE)
    best_fold = min(fold_results, key=lambda x: x['mse'])
    print(f"\nBest Fold: {best_fold['fold_num']}")
    print(f"  MSE: {best_fold['mse']:.6f}")
    print(f"  Pearson: {best_fold['pearson']:.4f}")
    
    # Plot feature importance for best model
    print(f"\nGenerating feature importance plot...")
    plot_feature_importance(best_fold['model'], feature_names, plots_dir, 
                           top_n=min(20, len(feature_names)))
    
    return {
        'fold_results': fold_results,
        'best_fold': best_fold,
        'avg_mse': avg_mse,
        'avg_mae': avg_mae,
        'avg_r2': avg_r2,
        'avg_pearson': avg_pearson,
        'plots_dir': plots_dir
    }


def main(config_path='config.yml', csv_path=None):
    """
    Main training function for Random Forest with 3-fold cross-validation
    
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
    
    # Get CSV path from config if not provided
    if csv_path is None:
        csv_path = config['data']['csv_path']
    
    print(f"\n{'='*60}")
    print(f"DATA PREPROCESSING")
    print(f"{'='*60}")
    
    # Process peptide data
    df_all, normalization_params = process_peptide_csv(
        csv_path, 
        normalize_target=config['random_forest'].get('normalize_target', True)
    )
    
    # Split data: train (80%) and test (20%)
    test_size = config['random_forest'].get('test_size', 0.2)
    random_seed = config['random_forest'].get('random_seed', 42)
    
    df_train, df_test = train_test_split(
        df_all, 
        test_size=test_size, 
        random_state=random_seed,
        shuffle=True
    )
    
    print(f"\nData Split:")
    print(f"  Training set: {len(df_train)} samples ({(1-test_size)*100:.0f}%)")
    print(f"  Test set: {len(df_test)} samples ({test_size*100:.0f}%)")
    
    # Extract features
    X_seq_train = df_train['descriptors_ifeat'].tolist()
    X_fix_train = np.vstack(df_train['descriptors'].tolist())
    y_train = np.array(df_train['Normalized_Potency'])
    
    X_seq_test = df_test['descriptors_ifeat'].tolist()
    X_fix_test = np.vstack(df_test['descriptors'].tolist())
    y_test = np.array(df_test['Normalized_Potency'])
    
    # Normalize fixed features
    scaler_type = config['random_forest'].get('scaler_type', 'StandardScaler')
    if scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler().fit(X_fix_train)
    else:
        scaler = StandardScaler().fit(X_fix_train)
    
    X_fix_train_norm = scaler.transform(X_fix_train)
    X_fix_test_norm = scaler.transform(X_fix_test)
    
    # Flatten features
    print(f"\nFlattening features...")
    X_train_flat, feature_names = flatten_features(X_seq_train, X_fix_train_norm)
    X_test_flat, _ = flatten_features(X_seq_test, X_fix_test_norm)
    
    print(f"  Total features: {X_train_flat.shape[1]}")
    
    # Perform K-fold cross-validation on training data
    n_folds = config['random_forest'].get('n_folds', 3)
    cv_results = k_fold_cross_validation(X_train_flat, y_train, feature_names, config, n_folds=n_folds)
    
    # Save results
    save_dir = Path(config['random_forest']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = cv_results['plots_dir']
    
    # Save scaler
    scaler_path = save_dir / 'rf_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\n✓ Saved scaler: {scaler_path}")
    
    # Save normalization parameters
    if normalization_params:
        norm_path = save_dir / 'rf_normalization_params.json'
        with open(norm_path, 'w') as f:
            json.dump(normalization_params, f, indent=2)
        print(f"✓ Saved normalization params: {norm_path}")
    
    # Get best model
    best_model = cv_results['best_fold']['model']
    
    # Evaluate on test set
    print(f"\n{'='*60}")
    print(f"EVALUATING ON TEST SET")
    print(f"{'='*60}")
    
    y_test_pred = best_model.predict(X_test_flat)
    
    # Calculate test metrics
    test_mse = np.mean((y_test - y_test_pred)**2)
    test_mae = np.mean(np.abs(y_test - y_test_pred))
    test_rmse = np.sqrt(test_mse)
    test_r2 = 1 - (np.sum((y_test - y_test_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
    test_pearson = pearsonr(y_test, y_test_pred)[0]
    
    print(f"\nTest Set Performance:")
    print(f"  MSE: {test_mse:.6f}")
    print(f"  RMSE: {test_rmse:.6f}")
    print(f"  MAE: {test_mae:.6f}")
    print(f"  R²: {test_r2:.6f}")
    print(f"  Pearson: {test_pearson:.4f}")
    
    # Plot test set predictions
    print(f"\nGenerating test set plot...")
    plot_predictions(
        y_true=y_test,
        y_pred=y_test_pred,
        pearson=test_pearson,
        fold_num='FINAL',
        save_dir=plots_dir,
        dataset_name='test'
    )
    
    # Save best model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"rf_best_model_{n_folds}fold_test_mse_{test_mse:.6f}_pearson_{test_pearson:.4f}_{timestamp}.pkl"
    model_path = save_dir / model_filename
    
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\n✓ Saved best model: {model_path}")
    
    # Save cross-validation summary
    cv_summary = {
        'model_type': 'RandomForest',
        'n_folds': n_folds,
        'n_estimators': config['random_forest'].get('n_estimators', 100),
        'max_depth': config['random_forest'].get('max_depth', None),
        'fold_results': [
            {
                'fold_num': r['fold_num'],
                'mse': float(r['mse']),
                'mae': float(r['mae']),
                'r2': float(r['r2']),
                'pearson': float(r['pearson'])
            }
            for r in cv_results['fold_results']
        ],
        'avg_mse': float(cv_results['avg_mse']),
        'avg_mae': float(cv_results['avg_mae']),
        'avg_r2': float(cv_results['avg_r2']),
        'avg_pearson': float(cv_results['avg_pearson']),
        'test_mse': float(test_mse),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'test_pearson': float(test_pearson),
        'best_fold_num': cv_results['best_fold']['fold_num'],
        'training_date': timestamp,
        'total_features': X_train_flat.shape[1]
    }
    
    summary_path = save_dir / f'rf_cv_summary_{timestamp}.json'
    with open(summary_path, 'w') as f:
        json.dump(cv_summary, f, indent=2)
    print(f"✓ Saved CV summary: {summary_path}")
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best model saved to: {model_path}")
    print(f"CV Average MSE: {cv_results['avg_mse']:.6f}")
    print(f"CV Average Pearson: {cv_results['avg_pearson']:.4f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test Pearson: {test_pearson:.4f}")
    print(f"\nAll plots saved to: {plots_dir}")
    print(f"  - Fold validation plots: rf_fold_X_validation_predictions.png")
    print(f"  - Test set plot: rf_fold_FINAL_test_predictions.png")
    print(f"  - CV summary plot: rf_cv_folds_summary.png")
    print(f"  - Feature importance: rf_feature_importance.png")

    return best_model, cv_results, {
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'test_pearson': test_pearson
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train Random Forest Regression Model with 3-Fold CV')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to peptide CSV file')
    args = parser.parse_args()
    
    main(args.config, csv_path=args.data)