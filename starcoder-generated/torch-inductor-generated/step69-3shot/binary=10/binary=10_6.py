
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 8)
 
    def forward(self, x1, x2):
        v3 = self.linear(x1)
        v4 = v4 + x2
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
x2 = torch.randn(1, 32)
