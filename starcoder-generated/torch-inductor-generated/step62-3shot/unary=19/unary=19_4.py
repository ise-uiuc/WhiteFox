
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(2, stride=2)
        self.linear = torch.nn.Linear(256, 1)
 
    def forward(self, x1):
        v1 = self.pool(x1)
        v2 = self.linear(v1)
        if v2 is None or v2 is not None and v2 > 0:
            v3 = v2
        else:
            v3 = 0
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256, 56, 56)
