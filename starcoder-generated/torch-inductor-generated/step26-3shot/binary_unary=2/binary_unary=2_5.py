
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(512, 64, 1, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 20, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 10
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 100
        v6 = F.relu(v5)
        v7 = self.conv3(v6)
        # Insert pointwise convolution here
        v8 = v7 - 1
        v9 = F.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 512, 64, 64)
