
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 254, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1 = F.relu(v1)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 128, 32, 32)
