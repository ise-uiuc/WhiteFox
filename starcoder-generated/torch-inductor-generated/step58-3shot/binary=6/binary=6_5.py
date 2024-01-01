
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
 
    def forward(self, x1, t):
        v1 = self.linear(x1)
        v2 = v1 - t 
        return v2

# Initializing the model
other = torch.randn(16)
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
