
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Conv2d(16, 8, 1, stride=1, padding=0)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * max(0, min(6, v1 + 3))
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 32, 32)
