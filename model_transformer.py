import torch
import torch.nn as nn # neural network module

# transformer encoder based: regression model
class TransformerRegressor(nn.Module):

    def __init__(self, seq_len, d_model = 64, nhead = 4, num_layers= 2):

        """
            transformer-based time series regression model

            args: 
                seq_len: input time window (24 hours)
                d_model: embedding size
                nhead: multihead number
                num_layers: transformer encoder layer number
        """
        super(TransformerRegressor, self).__init__()

        # input layer
        self.input_proj = nn.Linear(1, d_model)

        # transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model= d_model, nhead= nhead, batch_first= True) # (B, T, d)
        
        # transformer
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers= num_layers)

        # output layer
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, X):

        """
        forwardpropagation function
        args:
            X: input tensor

        return:
            out: predicted value (energy consumption value)  
        """
        
        X = self.input_proj(X)

        # pass through the transformer encoder layer 
        X = self.transformer(X)

        # only the last timestep output
        last_timestep = X[:, -1, :] # last timestep

        # prediction
        out = self.fc_out(last_timestep)

        return out
