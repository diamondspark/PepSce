"""
Random Forest Inference Utilities for Peptide Regression

This module provides utilities to load trained Random Forest models and
make predictions on peptide descriptors.
"""

import numpy as np
import pickle
from pathlib import Path

def load_rf_model(model_path):
    """
    Load a trained Random Forest model from pickle file
    
    Parameters
    ----------
    model_path : str or Path
        Path to the saved model pickle file
    
    Returns
    -------
    model : RandomForestRegressor
        Loaded Random Forest model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✓ Loaded Random Forest model from {model_path}")
    print(f"  - Number of trees: {model.n_estimators}")
    print(f"  - Number of features: {model.n_features_in_}")
    
    return model


def load_rf_scaler(scaler_path):
    """
    Load the fitted scaler for fixed-size features
    
    Parameters
    ----------
    scaler_path : str or Path
        Path to the saved scaler pickle file
    
    Returns
    -------
    scaler : sklearn scaler
        Fitted StandardScaler or MinMaxScaler
    """
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"✓ Loaded scaler from {scaler_path}")
    return scaler


def flatten_single_peptide_features(ifeat_features, modlamp_features):
    """
    Flatten features for a single peptide
    
    Parameters
    ----------
    ifeat_features : np.ndarray
        Sequence-based descriptors with shape (channels, seq_len)
    modlamp_features : np.ndarray
        Fixed-size descriptors with shape (n_features,)
    
    Returns
    -------
    np.ndarray
        Flattened feature vector with shape (1, total_features)
    """
    # Ensure numpy arrays
    if isinstance(ifeat_features, list):
        ifeat_features = np.array(ifeat_features)
    if isinstance(modlamp_features, list):
        modlamp_features = np.array(modlamp_features)
    
    # Flatten ifeat features
    ifeat_flat = ifeat_features.flatten()
    
    # Ensure modlamp is 1D
    if modlamp_features.ndim > 1:
        modlamp_flat = modlamp_features.flatten()
    else:
        modlamp_flat = modlamp_features
    
    # Concatenate
    features = np.concatenate([ifeat_flat, modlamp_flat])
    
    # Reshape to (1, n_features) for prediction
    return features.reshape(1, -1)


def get_Regression_Reward_RF(model, scaler, modlamp_features, ifeat_features):
    """
    Compute regression reward for a peptide using Random Forest model
    
    Parameters
    ----------
    model : RandomForestRegressor
        Trained Random Forest model
    scaler : sklearn scaler
        Fitted scaler for modlamp features
    modlamp_features : np.ndarray or list
        Fixed-size descriptor features (e.g., modlamp), shape: (n_features,)
    ifeat_features : np.ndarray or list
        Sequence-based descriptor features (e.g., ifeat), shape: (channels, seq_len)
    
    Returns
    -------
    float
        Regression prediction (scalar reward)
    
    Examples
    --------
    >>> # Load model and scaler
    >>> model = load_rf_model('./models/random_forest/rf_best_model.pkl')
    >>> scaler = load_rf_scaler('./models/random_forest/rf_scaler.pkl')
    >>> 
    >>> # Get descriptors for a peptide
    >>> modlamp = np.random.randn(139)  # Example modlamp features
    >>> ifeat = np.random.randn(14, 15)  # Example ifeat features (length 15)
    >>> 
    >>> # Get prediction
    >>> reward = get_Regression_Reward_RF(model, scaler, modlamp, ifeat)
    >>> print(f"Predicted reward: {reward:.4f}")
    """
    
    # Ensure numpy arrays
    if isinstance(modlamp_features, list):
        modlamp_features = np.array(modlamp_features)
    if isinstance(ifeat_features, list):
        ifeat_features = np.array(ifeat_features)
    
    # Normalize modlamp features
    if modlamp_features.ndim == 1:
        modlamp_normalized = scaler.transform(modlamp_features.reshape(1, -1))
    else:
        modlamp_normalized = scaler.transform(modlamp_features)
    
    # Flatten features
    features = modlamp_features #flatten_single_peptide_features(ifeat_features, modlamp_normalized[0])
    
    # Make prediction
    prediction = model.predict(features)
    
    return float(prediction[0])


class RandomForestPredictor:
    """
    Convenience class for making predictions with Random Forest model
    """
    
    def __init__(self, model_path, scaler_path):
        """
        Initialize predictor with model and scaler
        
        Parameters
        ----------
        model_path : str or Path
            Path to saved Random Forest model (.pkl)
        scaler_path : str or Path
            Path to saved scaler (.pkl)
        """
        self.model = load_rf_model(model_path)
        self.scaler = load_rf_scaler(scaler_path)
        self.n_features = self.model.n_features_in_
        
        print(f"\n✓ RandomForestPredictor initialized")
        print(f"  - Expected total features: {self.n_features}")
    
    def predict(self, modlamp_features, ifeat_features):
        """
        Make a prediction for a single peptide
        
        Parameters
        ----------
        modlamp_features : np.ndarray or list
            Fixed-size descriptors (e.g., modlamp)
        ifeat_features : np.ndarray or list
            Sequence-based descriptors (e.g., ifeat)
        
        Returns
        -------
        float
            Predicted regression score
        """
        return get_Regression_Reward_RF(
            self.model,
            self.scaler,
            modlamp_features,
            ifeat_features
        )

def find_best_rf_model(save_dir='./models/random_forest/', verbose=True):
    """
    Find the best Random Forest model in the save directory
    
    Parameters
    ----------
    save_dir : str or Path
        Directory where models are saved
    verbose : bool
        Whether to print information
    
    Returns
    -------
    dict
        Dictionary with 'model_path', 'scaler_path', and metrics
    """
    save_dir = Path(save_dir)
    
    # Find all model files
    model_files = list(save_dir.glob('rf_best_model_*.pkl'))
    
    if not model_files:
        raise FileNotFoundError(f"No Random Forest models found in {save_dir}")
    
    if verbose:
        print(f"\nFound {len(model_files)} model(s) in {save_dir}:")
    
    # Extract metrics from filenames and find best
    best_model = None
    best_pearson = -1
    model_info_list = []
    
    for model_path in model_files:
        try:
            # Parse filename: rf_best_model_3fold_test_mse_X_pearson_Y_timestamp.pkl
            filename = model_path.stem
            parts = filename.split('_')
            
            # Find pearson value
            pearson_idx = parts.index('pearson')
            pearson = float(parts[pearson_idx + 1])
            
            # Find mse value
            mse_idx = parts.index('mse')
            mse = float(parts[mse_idx + 1])
            
            model_info = {
                'path': model_path,
                'mse': mse,
                'pearson': pearson,
                'filename': model_path.name
            }
            model_info_list.append(model_info)
            
            if verbose:
                print(f"  [{model_path.name}]")
                print(f"    - MSE: {mse:.6f}")
                print(f"    - Pearson: {pearson:.4f}")
            
            if pearson > best_pearson:
                best_pearson = pearson
                best_model = model_info
        except:
            if verbose:
                print(f"  Warning: Could not parse {model_path.name}")
    
    if best_model is None:
        raise ValueError("Could not find valid model files")
    
    # Check for scaler
    scaler_path = save_dir / 'rf_scaler.pkl'
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    
    # Check for normalization params
    norm_files = list(save_dir.glob('rf_normalization_params.json'))
    norm_path = norm_files[0] if norm_files else None
    
    result = {
        'model_path': best_model['path'],
        'scaler_path': scaler_path,
        'normalization_path': norm_path,
        'mse': best_model['mse'],
        'pearson': best_model['pearson'],
        'all_models': model_info_list
    }
    
    if verbose:
        print(f"\n✓ Best model (highest Pearson):")
        print(f"  Path: {best_model['path']}")
        print(f"  MSE: {best_model['mse']:.6f}")
        print(f"  Pearson: {best_model['pearson']:.4f}")
    
    return result


# ==================== Usage Examples ====================

if __name__ == "__main__":
    print("="*70)
    print("RANDOM FOREST INFERENCE EXAMPLES")
    print("="*70)
    
    # Example 0: Find available models
    print("\n" + "="*70)
    print("Example 0: Finding Available Models")
    print("="*70)
    
    try:
        model_info = find_best_rf_model('./models/random_forest/', verbose=True)
        print(f"\nRecommended paths for loading:")
        print(f"  Model: {model_info['model_path']}")
        print(f"  Scaler: {model_info['scaler_path']}")
    except FileNotFoundError as e:
        print(f"Note: {e}")
        print("Please train a model first using train_random_forest.py")
    
    # Example 1: Direct function usage
    print("\n" + "="*70)
    print("Example 1: Direct Function Usage")
    print("="*70)
    
    try:
        model = load_rf_model('./models/random_forest/rf_best_model.pkl')
        scaler = load_rf_scaler('./models/random_forest/rf_scaler.pkl')
        
        # Dummy features for demonstration
        modlamp = np.random.randn(139)
        ifeat = np.random.randn(14, 15)  # 14 channels, seq_len=15
        
        prediction = get_Regression_Reward_RF(model, scaler, modlamp, ifeat)
        print(f"\nPrediction: {prediction:.4f}")
    except FileNotFoundError:
        print("Model not found. Skipping this example.")
    
