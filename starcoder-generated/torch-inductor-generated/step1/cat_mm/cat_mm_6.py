
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(3, 8, 7, stride=2, padding=3)
 
    def forward(self, x):
        v4 = self.conv3(x)
        v5 = self.conv2(x)
        v6 = self.conv1(x)
        v7 = torch.cat([v4, v5, v6], dim=1)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
