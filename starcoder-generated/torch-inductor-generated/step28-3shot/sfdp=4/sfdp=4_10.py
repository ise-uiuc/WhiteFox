
class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.multihead_attention = torch.nn.MultiheadAttention(input_dim, num_heads)
 
    def forward(self, query, key, value, attn_mask):
        return self.multihead_attention(query, key, value, attn_mask)

# Initializing the model
config = {'d_model': 3, 'd_k': 3, 'd_v': 3, 'num_heads': 3}
m = Model(config)

# Inputs to the model
query = torch.randn(1, 2, 3)
key = value = torch.randn(1, 3, 3)
vmask = torch.ones(size=(1, 2, 3)) # attention mask
