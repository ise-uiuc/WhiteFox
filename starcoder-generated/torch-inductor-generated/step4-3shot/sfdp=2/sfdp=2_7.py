
class Model(torch.nn.Module):
    def __init__(self, query_channels, key_channels, n_units, n_heads, dropout):
        super().__init__()
        self.attn = AttentionCore(query_channels, key_channels, n_units, n_heads, dropout)
 
    def forward(self, query, key, value, scale_factor=1, dropout_p=0.2):
        v = self.attn(query, key, value, scale_factor, dropout_p)
        return v

# Initializing the model
m = Model(query_channels=16, key_channels=16, n_units=16, n_heads=3, dropout=0.2)

# Inputs to the model
query = torch.randn(1, 32, 5, 32)
key = torch.randn(1, 32, 4, 32)
value = torch.randn(1, 32, 5, 32)
scale_factor = 100
dropout_p = 0.3
