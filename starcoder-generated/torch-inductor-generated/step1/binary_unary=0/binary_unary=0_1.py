
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0)
 
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        r1 = torch.relu(v1)
        return r1 + v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
