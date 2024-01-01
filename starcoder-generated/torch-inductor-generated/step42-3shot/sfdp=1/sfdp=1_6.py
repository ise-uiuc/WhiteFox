
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
 
    def forward(self, query, key, value, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        return torch.matmul(scaled_qk.softmax(dim=-1), value)

# Initializing the model
m = ScaledDotProductAttention(math.sqrt(0.5))

# Inputs to the model
query = torch.randn(1, 16, 64)
key = torch.randn(1, 16, 64)
value = torch.randn(1, 16, 64)
dropout_p = 0.2
