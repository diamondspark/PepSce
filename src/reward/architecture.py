import torch
import torch.nn as nn

class CNN1D(nn.Module):
    """
    1D CNN for processing sequence-based descriptors (e.g., ifeat)
    Automatically adapts to input sequence length
    """
    def __init__(self, in_channels=14, out_channels=5, kernel_sizes=[2, 3, 4, 5], 
                 fc_hidden_dims=[735, 128, 64]):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels/features per position
        out_channels : int
            Number of output channels for each conv layer
        kernel_sizes : list of int
            Kernel sizes for parallel convolutional branches
        fc_hidden_dims : list of int
            Hidden dimensions for fully connected layers
        """
        super(CNN1D, self).__init__()
        
        self.kernel_sizes = kernel_sizes
        self.out_channels = out_channels
        
        # Create parallel convolutional branches
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=k)
            for k in kernel_sizes
        ])
        
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(out_channels)
            for _ in kernel_sizes
        ])
        
        self.activation = nn.LeakyReLU(inplace=False)
        
        # Pooling layers (pool size matches kernel size)
        self.pool_layers = nn.ModuleList([
            nn.MaxPool1d(k) for k in kernel_sizes
        ])
        
        # Fully connected layers (input size computed dynamically)
        self.fc_layers = None  # Will be initialized in first forward pass
        self.fc_hidden_dims = fc_hidden_dims
        self._initialized = False
        
    def _initialize_fc_layers(self, flattened_size):
        """Initialize FC layers based on the actual flattened size"""
        layers = []
        prev_dim = flattened_size
        
        for hidden_dim in self.fc_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(inplace=False)
            ])
            prev_dim = hidden_dim
        
        self.fc_layers = nn.Sequential(*layers)
        self._initialized = True
        
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape: (batch_size, in_channels, seq_len)
        
        Returns
        -------
        torch.Tensor
            Shape: (batch_size, fc_hidden_dims[-1])
        """
        # Parallel convolutional branches
        conv_outputs = []
        for conv, bn, pool in zip(self.conv_layers, self.bn_layers, self.pool_layers):
            out = self.activation(bn(conv(x)))
            out = pool(out)
            conv_outputs.append(out)
        
        # Concatenate along feature dimension
        union = torch.cat(conv_outputs, dim=2)
        union = union.reshape(union.size(0), -1)
        
        # Initialize FC layers on first forward pass
        if not self._initialized:
            self._initialize_fc_layers(union.size(1))
            # Move FC layers to same device as input
            self.fc_layers = self.fc_layers.to(x.device)
        
        # Pass through FC layers
        output = self.fc_layers(union)
        
        return output


class MLP(nn.Module):
    """
    MLP for processing fixed-size descriptors (e.g., modlamp)
    """
    def __init__(self, input_dim=139, hidden_dims=[100, 64], use_batchnorm=True):
        """
        Parameters
        ----------
        input_dim : int
            Input feature dimension
        hidden_dims : list of int
            Hidden layer dimensions
        use_batchnorm : bool
            Whether to use batch normalization
        """
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape: (batch_size, input_dim)
        
        Returns
        -------
        torch.Tensor
            Shape: (batch_size, hidden_dims[-1])
        """
        return self.network(x)


class PeptideRegressionModel(nn.Module):
    """
    Combined model for peptide regression
    Handles variable-length sequence descriptors (CNN) and fixed-size descriptors (MLP)
    """
    def __init__(self, 
                 # CNN parameters
                 cnn_in_channels=14,
                 cnn_out_channels=5,
                 cnn_kernel_sizes=[2, 3, 4, 5],
                 cnn_fc_dims=[735, 128, 64],
                 # MLP parameters
                 mlp_input_dim=139,
                 mlp_hidden_dims=[100, 64],
                 mlp_use_batchnorm=True,
                 # Fusion parameters
                 fusion_hidden_dims=[64, 32],
                 dropout=0.0):
        """
        Parameters
        ----------
        cnn_in_channels : int
            Number of channels in sequence descriptors
        cnn_out_channels : int
            Number of output channels for CNN layers
        cnn_kernel_sizes : list of int
            Kernel sizes for parallel CNN branches
        cnn_fc_dims : list of int
            Hidden dimensions for CNN's fully connected layers
        mlp_input_dim : int
            Dimension of fixed-size descriptors
        mlp_hidden_dims : list of int
            Hidden dimensions for MLP
        mlp_use_batchnorm : bool
            Whether to use batch normalization in MLP
        fusion_hidden_dims : list of int
            Hidden dimensions for fusion layers
        dropout : float
            Dropout probability (0 = no dropout)
        """
        super(PeptideRegressionModel, self).__init__()
        
        self.cnn = CNN1D(
            in_channels=cnn_in_channels,
            out_channels=cnn_out_channels,
            kernel_sizes=cnn_kernel_sizes,
            fc_hidden_dims=cnn_fc_dims
        )
        
        self.mlp = MLP(
            input_dim=mlp_input_dim,
            hidden_dims=mlp_hidden_dims,
            use_batchnorm=mlp_use_batchnorm
        )
        
        # Fusion layers
        # Input dimension is CNN output + MLP output
        fusion_input_dim = cnn_fc_dims[-1] + mlp_hidden_dims[-1]
        
        fusion_layers = []
        prev_dim = fusion_input_dim
        
        for hidden_dim in fusion_hidden_dims:
            fusion_layers.append(nn.Linear(prev_dim, hidden_dim))
            fusion_layers.append(nn.ReLU())
            if dropout > 0:
                fusion_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (regression - single value)
        fusion_layers.append(nn.Linear(prev_dim, 1))
        
        self.fusion = nn.Sequential(*fusion_layers)
        
    def forward(self, X_sequence, X_fixed):
        """
        Parameters
        ----------
        X_sequence : torch.Tensor
            Sequence-based descriptors, shape: (batch_size, channels, seq_len)
        X_fixed : torch.Tensor
            Fixed-size descriptors, shape: (batch_size, feature_dim)
        
        Returns
        -------
        torch.Tensor
            Regression predictions, shape: (batch_size, 1)
        """
        cnn_output = self.cnn(X_sequence)
        mlp_output = self.mlp(X_fixed)
        
        # Concatenate outputs
        concat = torch.cat((cnn_output, mlp_output), dim=-1)
        
        # Final prediction
        output = self.fusion(concat)
        
        return output
    
    def get_architecture_summary(self):
        """Get a summary of the model architecture"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'cnn_output_dim': self.cnn.fc_hidden_dims[-1],
            'mlp_output_dim': self.mlp.output_dim,
        }

