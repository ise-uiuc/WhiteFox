
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, attn_mask):
        v = torch.matmul(query, key.transpose(-2, -1))
        v = v / math.sqrt(query.size(-1))
        v = v + attn_mask
        v = torch.softmax(v, dim=-1)
        v = torch.matmul(v, value)
        return v

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 10, 16)
key = torch.randn(1, 10, 16)
value = torch.randn(1, 10, 32)
attn_mask = torch.randn(1, 10, 10)
