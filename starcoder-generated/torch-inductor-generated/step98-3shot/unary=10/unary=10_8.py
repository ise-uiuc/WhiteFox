
class Model(torch.nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(6, out_features)
 
    def forward(self, x5):
        x1 = self.linear(x5)
        x2 = x1 + 3
        x3 = torch.clamp_min(x2, 0)
        x4 = torch.clamp_max(x3, 6)
        x6 = x4 / 6
        return x6

# Initializing the model
m = Model(8)

# Inputs to the model
x5 = torch.randn(1, 6)
