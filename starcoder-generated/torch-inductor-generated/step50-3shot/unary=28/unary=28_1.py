
class Model(torch.nn.Module):
    def __init__(self, *, min_value=0, max_value=10):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=3, out_features=1)

    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3

# Intializing the model
m = Model(min_value=0, max_value=10)

# Inputs to the model
x = torch.randn(1, 3)
