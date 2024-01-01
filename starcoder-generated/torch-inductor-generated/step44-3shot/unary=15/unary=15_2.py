
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(512, 1024, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1024, 2048, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = v4.view(x1.size(0), -1)
        return v5
# Inputs to the model
x1 = torch.randn(1, 512, 32, 32)
