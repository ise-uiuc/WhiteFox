
class Model(torch.nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.linear = torch.nn.Linear(20, dim)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        return v1 + x2

# Initializing the model
dim = 256
m = Model(dim)

# Inputs to the model
x1 = torch.randn(4, 20)
x2 = torch.randn(4, dim)
