
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 6)
 
    def forward(self, x1):
        x1 = x1.flatten(-2, -1)
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
__model__

# Inputs to the model
x1 = torch.randn(1, 32, 128, 128)
other = torch.randn(1, 6)
