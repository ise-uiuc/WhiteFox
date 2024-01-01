
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x0):
        v0 = self.linear(x0)
        return v0 + torch.randn(1, 32)

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 16)
