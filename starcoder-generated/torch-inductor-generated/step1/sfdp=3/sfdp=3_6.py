
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.linear1 = torch.nn.Linear(960, 800)
        self.linear2 = torch.nn.Linear(800, 8)
 
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = torch.flatten(v3, start_dim=1)
        v5 = self.linear1(v4)
        v6 = self.linear2(v5)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
