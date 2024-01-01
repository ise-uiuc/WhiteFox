
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Linear(16, 32)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + torch.rand_like(v1)
        v3 = v2.relu()
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 4, 4)
