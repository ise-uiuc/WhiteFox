
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2304, 2304, bias=False)
        self.bn = torch.nn.BatchNorm1d(2304)
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        v10 = x1.view(-1, 2304)
        v1 = self.fc1(v10)
        v2 = self.bn(v1)
        v3 = v2.view(int(v2.size(0)), int(32.0), 8, 4)
        v4 = self.conv(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2304)
