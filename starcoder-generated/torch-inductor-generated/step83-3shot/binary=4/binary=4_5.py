
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
    
    def forward(self, x1, x2, other=None):
        x = torch.cat([x1, x2], dim=1)
        if other is not None:
            x = x + other
        x = self.linear(x)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 8)
x2 = torch.randn(1, 8, 8)
_other = torch.randn(1, 8, 8)
