
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Linear(3, 4)
 
    def forward(self, x1, x2=None):
        v1 = self.conv(x1)
        if not x2 is None:
            v2 = v1 + x2
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3)
x2 = torch.randn(2, 3)
