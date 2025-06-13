import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models.spectral_convolution import SpectralConv1d


class SpectralConvLSTM(nn.Module):
    """
    A model that combines spectral convolution from Fourier Neural Operator (FNO)
    with LSTM for time-series data processing.
    
    The model first applies spectral convolution to the time series data and 
    then feeds the processed data into an LSTM network.
    
    Args:
        input_dim (int): Input dimension of the time series data
        hidden_channels (int): Number of hidden channels in spectral convolution
        n_modes (int): Number of Fourier modes to keep in spectral convolution
        hidden_dim (int): Hidden dimension of the LSTM
        layer_dim (int): Number of LSTM layers
        output_dim (int): Output dimension
        dropout (float, optional): Dropout probability. Default: 0.0
    """
    def __init__(self, input_dim, hidden_channels, n_modes, hidden_dim, layer_dim, output_dim, dropout=0.0):
        super(SpectralConvLSTM, self).__init__()
        
        # Spectral convolution parameters
        self.hidden_channels = hidden_channels
        self.n_modes = n_modes
        
        # LSTM parameters
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        # Input projection
        self.fc0 = nn.Linear(input_dim, hidden_channels)
        
        # Spectral convolution layer
        self.spectral_conv = SpectralConv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            modes=n_modes
        )
        
        # Additional processing after spectral convolution
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True
        )
        
        # Output layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, h0=None, c0=None):
        """
        Forward pass of the SpectralConvLSTM model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim]
            h0 (torch.Tensor, optional): Initial hidden state for LSTM
            c0 (torch.Tensor, optional): Initial cell state for LSTM
            
        Returns:
            out (torch.Tensor): Output tensor of shape [batch_size, output_dim]
            hn (torch.Tensor): Final hidden state
            cn (torch.Tensor): Final cell state
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize LSTM states if not provided
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(x.device)
        
        # Process each time step with spectral convolution
        spectral_outputs = []
        
        for t in range(seq_len):
            # Get current time step
            x_t = x[:, t, :]
            
            # Project input to hidden dimension
            x_t = self.fc0(x_t)  # [batch_size, hidden_channels]
            
            # Reshape for spectral convolution (needs [batch, channel, spatial_dim])
            x_t = x_t.unsqueeze(-1)  # [batch_size, hidden_channels, 1]
            
            # Apply spectral convolution
            x_t = self.spectral_conv(x_t)  # [batch_size, hidden_channels, 1]
            
            # Reshape back
            x_t = x_t.squeeze(-1)  # [batch_size, hidden_channels]
            
            # Additional processing
            x_t = F.gelu(self.fc1(x_t))
            
            # Apply dropout
            x_t = self.dropout(x_t)
            
            # Collect output for this time step
            spectral_outputs.append(x_t)
        
        # Stack outputs along sequence dimension
        spectral_out = torch.stack(spectral_outputs, dim=1)  # [batch_size, seq_len, hidden_channels]
        
        # Pass through LSTM
        lstm_out, (hn, cn) = self.lstm(spectral_out, (h0, c0))
        
        # Get output from the last time step
        out = self.fc2(lstm_out[:, -1, :])
        
        return out, hn, cn


class FNO1dLSTM(nn.Module):
    """
    A more complete implementation of FNO1d combined with LSTM for time-series data.
    This implementation follows the standard FNO architecture with multiple spectral layers
    before feeding into an LSTM network.
    
    Args:
        modes (int): Number of Fourier modes to keep
        width (int): Number of channels in the convolutional layers
        input_dim (int): Input dimension of the time series data
        hidden_dim (int): Hidden dimension of the LSTM
        layer_dim (int): Number of LSTM layers
        output_dim (int): Output dimension
        n_layers (int): Number of spectral convolution layers
    """
    def __init__(self, modes, width, input_dim, hidden_dim, layer_dim, output_dim, n_layers=4):
        super(FNO1dLSTM, self).__init__()
        
        self.modes = modes
        self.width = width
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.n_layers = n_layers
        
        # Input projection
        self.fc0 = nn.Linear(input_dim, self.width)
        
        # Spectral convolution layers
        self.spectral_layers = nn.ModuleList([])
        for i in range(n_layers):
            self.spectral_layers.append(
                SpectralConv1d(
                    in_channels=self.width,
                    out_channels=self.width,
                    modes=self.modes
                )
            )
            
        # Linear layers after each spectral convolution
        self.linear_layers = nn.ModuleList([])
        for i in range(n_layers):
            self.linear_layers.append(nn.Linear(self.width, self.width))
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.width,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True
        )
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, h0=None, c0=None):
        """
        Forward pass of the FNO1dLSTM model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim]
            h0 (torch.Tensor, optional): Initial hidden state for LSTM
            c0 (torch.Tensor, optional): Initial cell state for LSTM
            
        Returns:
            out (torch.Tensor): Output tensor of shape [batch_size, output_dim]
            hn (torch.Tensor): Final hidden state
            cn (torch.Tensor): Final cell state
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize LSTM states if not provided
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(x.device)
        
        # Process each time step with FNO
        fno_outputs = []
        
        for t in range(seq_len):
            # Get current time step
            x_t = x[:, t, :]
            
            # Project input to hidden dimension
            x_t = self.fc0(x_t)  # [batch_size, width]
            
            # Reshape for spectral convolution (needs [batch, channel, spatial_dim])
            x_t = x_t.unsqueeze(-1)  # [batch_size, width, 1]
            
            # Apply spectral convolution layers
            for i in range(self.n_layers):
                # Spectral convolution
                x1 = self.spectral_layers[i](x_t)
                
                # Linear transformation
                x2 = self.linear_layers[i](x_t.squeeze(-1)).unsqueeze(-1)
                
                # Combine and apply activation
                x_t = F.gelu(x1 + x2)
            
            # Reshape back
            x_t = x_t.squeeze(-1)  # [batch_size, width]
            
            # Collect output for this time step
            fno_outputs.append(x_t)
        
        # Stack outputs along sequence dimension
        fno_out = torch.stack(fno_outputs, dim=1)  # [batch_size, seq_len, width]
        
        # Pass through LSTM
        lstm_out, (hn, cn) = self.lstm(fno_out, (h0, c0))
        
        # Get output from the last time step
        out = self.fc_out(lstm_out[:, -1, :])
        
        return out, hn, cn