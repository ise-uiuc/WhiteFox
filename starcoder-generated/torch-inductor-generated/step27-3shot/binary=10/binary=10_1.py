
class Model(torch.nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(32, out_features)
 
    def forward(self, x1, other=None):
        v1 = self.linear(x1)
        if other is None:
            v1 = v1 + torch.abs(x1)
        else:
            v1 = v1 + other
        return v1

# Initializing the model
m = Model(8)

# Inputs to the model
x1 = torch.randn(1, 32)
