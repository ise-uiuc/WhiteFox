
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(3, 8)
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        v0 = x1
        v2 = v0
        v3 = self.net(v2)
        v4 = torch.sigmoid(v3)
        v5 = v4
        v6 = self.conv(v5)
        v7 = v6
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
