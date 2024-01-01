
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Linear(16, 32)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        t2 = v1 + a
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
a = torch.randn(3, 1)
