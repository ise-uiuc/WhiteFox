
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(256, 1, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.relu(v2)
        v4 = v3 + x1
        return v4
# Inputs to the model
x1 = torch.randn(1, 256, 7, 7)
