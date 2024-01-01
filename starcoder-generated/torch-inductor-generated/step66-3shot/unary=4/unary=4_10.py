
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        in_features = 20
        out_features = 30
        self.linear = torch.nn.Linear(in_features, out_features)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.8414709848078965
        v4 = torch.erf(v3)
        v5 = v4 + 2
        v6 = v2 * v5
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20)
