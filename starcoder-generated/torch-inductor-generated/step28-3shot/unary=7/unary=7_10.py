
class Model(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
 
    def forward(self, x1):
        x2 = self.linear(x1)
        x3 = torch.clamp(x2 + 3, min=0, max=6)
        x4 = x3 / 6
        return x4

# Initializing the model
m = Model(10, 10)

# Inputs to the model
x1 = torch.randn(1, 10)
