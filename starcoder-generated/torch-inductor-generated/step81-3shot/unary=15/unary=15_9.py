
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 13, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(13, 13, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(256, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv3(v1)
        # v3 is undefined
        v4 = torch.relu(v1)
        v5 = torch.relu(v2)
        v6 = torch.mul(v4, v5)
        return v6
# Inputs to the model
x1 = torch.randn(2, 2, 224, 224)
