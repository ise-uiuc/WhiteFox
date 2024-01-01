
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 1, stride=1, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.MaxPool2d(128)
        )
        self.linear = torch.nn.Linear(8, 4096)
        self.linear2 = torch.nn.Linear(4096, 4096)
   
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.linear(v1)
        v3 = self.linear2(v2)
        v4 = v3 + v1
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
