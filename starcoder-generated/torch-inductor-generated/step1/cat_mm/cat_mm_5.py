
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 3, 3, stride=2, padding=1)
 
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
#         v4 = v3.flatten(1) # for MLP
        v4 = torch.flatten(v3, 1) # for CNN
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
