
class Model(torch.nn.Module):
    def __init__(self, q, k, v, inv_scale):
        super().__init__()
        self.q = q
        self.k = k
        self.v = v
        self.inv_scale = inv_scale
 
    def forward(self, query, key, value):
        attention_weights = torch.matmul(query, key.transpose(-2, -1))
        scaled_attention_weights = attention_weights / self.inv_scale
        scaled_attention_weights = scaled_attention_weights.softmax(dim=-1)
        return scaled_attention_weights.matmul(value)

# Initializing the model
q = torch.randn(1, 12, 384)
k = torch.randn(1, 12, 384)
v = torch.randn(1, 12, 512)
m = Model(q, k, v, inv_scale=384**-0.5)

# Inputs to the model
query = torch.randn(1, 10, 384)
key = torch.randn(1, 10, 384)
value = torch.randn(1, 10, 512)
