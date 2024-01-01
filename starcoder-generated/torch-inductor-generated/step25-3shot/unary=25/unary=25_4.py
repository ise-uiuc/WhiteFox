
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 128)
        self.negative_slope = 0.1
 
    def forward(self, x1):
        v1 = self.linear(x1)
        s = self.negative_slope
        m = s * (v1 > 0)
        v2 = (m * v1) + ((1 - m) * s * v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 64)
