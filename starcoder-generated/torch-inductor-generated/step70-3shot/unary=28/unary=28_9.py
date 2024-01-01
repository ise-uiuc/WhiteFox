
class Model(nn.Module):
    def __init__(self, min_value=None, max_value=None):
        super().__init__()
        self.linear = nn.Linear(3, 8)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp(v1, min=min_value)
        v3 = torch.clamp(v2, max=max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
