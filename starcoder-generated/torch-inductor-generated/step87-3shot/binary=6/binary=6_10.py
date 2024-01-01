
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 9)
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 1, stride=1, padding=1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = self.conv1(x1)
        v3 = self.conv2(v2)
        v4 = v3 - v1
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
