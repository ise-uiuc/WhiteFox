
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x0, x1):
        v0 = self.linear(x0)
        v1 = v0 + x1
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 3, 64, 64)
x1 = torch.randn(1, 3, 64, 64)
