
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, other=None):
        x4 = self.linear(x1)
        if other is not None:
            x4 += other
        x5 = x4.relu()
        return x5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 48, 64)
