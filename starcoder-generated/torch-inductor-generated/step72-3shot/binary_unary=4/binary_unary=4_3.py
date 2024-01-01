
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x1, other=None):
        v1 = self.linear(x1)
        if other is None:
            other = torch.randn(2, 32)
        v2 = v1 * 0.5
        v3 = v2 + 1
        v4 = v2 * v3
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
