
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(64, 32)
        self.linear2 = torch.nn.Linear(32, 16)
 
    def forward(self, x1, x2):
        v1 = self.linear1(x1)
        v2 = self.linear2(v1)
        v3 = v2 - x2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
x2 = (torch.rand(1, 16) - 0.5) * 64
