
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Linear(16, 8)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 16)
