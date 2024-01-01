
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1.size()
        v3 = torch.FloatTensor(3, 3).uniform_(-0.5, 1.0)
        v4 = torch.matmul(v1, v3)
        return v2, {"inp": v4}

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
