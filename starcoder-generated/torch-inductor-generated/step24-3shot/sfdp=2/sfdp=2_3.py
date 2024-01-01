
class Model(torch.nn.Module):
    def __init__(self, embed, n_head, window_size=64, dropout_p=0.1, num_layers=1, heads_share_parameters=False):
        super().__init__()
        encoder_layers = TransformerEncoderLayer(embed, n_head, window_size=window_size, dropout_p=dropout_p, heads_share_parameters=heads_share_parameters, mode='dot_product')
        self.transformer = TransformerEncoder(encoder_layers, num_layers=num_layers)
 
    def forward(self, x1):
        v1 = self.transformer(x1, x1, x1)
        return v1

# Initializing the model
m = Model(embed, window_size=window_size, num_heads=num_heads, dropout_p=dropout_p, num_layers=num_layers)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
