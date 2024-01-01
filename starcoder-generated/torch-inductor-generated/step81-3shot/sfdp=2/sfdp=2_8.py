
class LinearMultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, d_feature, dropout_p):
      ...

    def forward(self, query, key, value, mask=None):
      ...

# Initializing the model
m = LinearMultiHeadAttention(8, 128, 0.1)

# Inputs to the model
query = torch.randn(1, 8, 256)
key = torch.randn(1, 8, 1024)
value = torch.randn(1, 8, 1024)
mask = torch.randn(1, 1, 1, 1024)
