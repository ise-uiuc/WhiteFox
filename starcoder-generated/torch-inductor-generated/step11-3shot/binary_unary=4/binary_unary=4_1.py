
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fused = torch.nn.Linear(8, 32)
 
    def forward(self, x1, other=None):
        v1 = self.fused(x1)
        v2 = v1 + other
        v3 = v2.relu()
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
other = torch.randn(1, 32)
