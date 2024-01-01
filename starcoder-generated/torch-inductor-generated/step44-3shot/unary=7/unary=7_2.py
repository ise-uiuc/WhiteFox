
class Model(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.linear = torch.nn.Linear(self.n_features, 2)
        self.linear_clamp = torch.nn.Linear(self.n_features, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp(self.linear_clamp(x1), min=0, max=6)
        v3 = v2 + 3
        v4 = v1 * v3
        return v4 / 6

# Initializing the model
m = Model(128)

# Inputs to the model
x1 = torch.randn(1, 128)
