
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Linear(32, 64, bias=False)

    def forward(self, queries, values, inv_scale):
        keys = self.key(queries)
        return torch.matmul(queries, keys.transpose(-2, -1)) * inv_scale

# Initializing the model
m = Model()

# Inputs to the model
queries = torch.randn(1, 64, 32)
values = torch.randn(1, 64, 32)
inv_scale = math.sqrt(0.5)
