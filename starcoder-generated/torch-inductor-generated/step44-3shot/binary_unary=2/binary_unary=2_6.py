
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(6, 16, 7, stride=1, padding=3, dilation=2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=2, padding=1, dilation=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0.5
        v3 = F.relu(v2)
        return self.conv2(v3)
# Inputs to the model
x1 = torch.randn(1, 6, 224, 224)
