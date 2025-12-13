
from torch.utils.data import Dataset
from dataset.descriptors import get_modlamp_descriptors, get_ifeat_desc
import pandas as pd

def process_peptide_csv(csv_path, normalize_target=True):
    """
    Process peptide CSV and calculate descriptors
    
    Parameters
    ----------
    csv_path : str
        Path to CSV with columns: Peptide Name, Sequence, Potency
    normalize_target : bool
        Whether to normalize potency values
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added descriptor columns
    """
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ['Peptide Name', 'Sequence', 'Potency']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")
    
    print(f"Processing {len(df)} peptides from {csv_path}")
    print(f"Calculating descriptors...")
    
    # Calculate descriptors for each peptide
    ifeat_list = []
    modlamp_list = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  Processed {idx}/{len(df)} peptides")
        
        sequence = row['Sequence']
        ifeat, modlamp = get_ifeat_desc(sequence), get_modlamp_descriptors(sequence)
        ifeat_list.append(ifeat)
        modlamp_list.append(modlamp)
    
    df['descriptors_ifeat'] = ifeat_list
    df['descriptors'] = modlamp_list
    
    # Normalize potency if requested
    if normalize_target:
        potency_mean = df['Potency'].mean()
        potency_std = df['Potency'].std()
        df['Normalized_Potency'] = (df['Potency'] - potency_mean) / potency_std
        
        # Save normalization parameters
        normalization_params = {
            'mean': float(potency_mean),
            'std': float(potency_std)
        }
        print(f"\nPotency normalization:")
        print(f"  Mean: {potency_mean:.4f}")
        print(f"  Std: {potency_std:.4f}")
    else:
        df['Normalized_Potency'] = df['Potency']
        normalization_params = None
    
    print(f"✓ Descriptor calculation complete")
    
    return df, normalization_params



class PeptideDataset(Dataset):
    """
    Dataset for peptide descriptors
    Handles both sequence-based and fixed-size descriptors
    """
    def __init__(self, X_sequence, X_fixed, y, sequence_transform=None, fixed_transform=None):
        """
        Parameters
        ----------
        X_sequence : list or np.ndarray
            Sequence-based descriptors (e.g., ifeat)
            Each element should have shape (channels, seq_len)
        X_fixed : list or np.ndarray
            Fixed-size descriptors (e.g., modlamp)
            Each element should have shape (feature_dim,)
        y : np.ndarray
            Target values
        sequence_transform : callable, optional
            Transform to apply to sequence descriptors
        fixed_transform : callable, optional
            Transform to apply to fixed descriptors
        """
        self.sequence_features = X_sequence
        self.fixed_features = X_fixed
        self.labels = y
        self.sequence_transform = sequence_transform
        self.fixed_transform = fixed_transform
        
    def __getitem__(self, index):
        seq_feat = self.sequence_features[index]
        fixed_feat = self.fixed_features[index]
        label = self.labels[index]
        
        # Apply transforms if provided
        if self.sequence_transform:
            seq_feat = self.sequence_transform(seq_feat)
        if self.fixed_transform:
            fixed_feat = self.fixed_transform(fixed_feat)
        
        return [seq_feat, fixed_feat], label
    
    def __len__(self):
        return len(self.labels)