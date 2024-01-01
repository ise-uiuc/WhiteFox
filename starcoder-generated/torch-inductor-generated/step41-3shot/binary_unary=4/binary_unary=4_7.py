
class Model(torch.nn.Module):
    def __init__(self, n_features, n_label=2):
        super().__init__()
        self.linear = torch.nn.Linear(20, n_label)
 
    def forward(self, x1, other=None):
        v1 = self.linear(x1)
        if other is not None:
            v1 = v1 + other
        return torch.nn.functional.relu(v1)

# Initializing the model
m = Model(n_features=20)

# Inputs to the model
x1 = torch.randn(1, 20)
