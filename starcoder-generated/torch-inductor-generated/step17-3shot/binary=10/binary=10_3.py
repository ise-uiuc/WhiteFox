
class Model(torch.nn.Module):
    def __init__(self, other, in_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model(other, in_features)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
