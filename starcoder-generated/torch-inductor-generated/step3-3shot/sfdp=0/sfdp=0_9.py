
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale):
        query = query.squeeze(1)
        key = key.squeeze(1)
        value = value.squeeze(1)
        scaled_dot_product = torch.matmul(query, key.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        return attention_weights.matmul(value)

# Initializing the model
m = ScaledDotProductAttention()
query = torch.randn(1, 1, 256)
key = torch.randn(1, 1, 256)
value = torch.randn(1, 1, 256)
inv_scale = 1 / math.sqrt(256)
