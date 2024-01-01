
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, inv_scale):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1 / inv_scale
        v3 = v2.softmax(dim=-1)
        v4 = v3.matmul(value)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 25)
key = torch.randn(1, 3, 20)
value = torch.randn(1, 3, 20)
__inv_scale__ = 1.0 / math.sqrt(query.shape[-1])
