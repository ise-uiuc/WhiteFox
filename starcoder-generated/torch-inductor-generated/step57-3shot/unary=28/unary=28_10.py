
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(10, 20)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=-2.0)
        v3 = torch.clamp_max(v2, max_value=10.0)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10)
