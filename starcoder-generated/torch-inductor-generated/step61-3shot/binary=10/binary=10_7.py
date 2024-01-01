
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Linear(13, 8)
 
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        if other is not None:
            v1 = v1 + other
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 13)
x2 = torch.randn(3, 13)
