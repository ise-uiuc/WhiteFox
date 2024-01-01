
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(64, 1, 1, stride=1)
 
    def forward(self, x1):
        v1 = self.conv1(x1)
        v3 = self.conv2(v1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
