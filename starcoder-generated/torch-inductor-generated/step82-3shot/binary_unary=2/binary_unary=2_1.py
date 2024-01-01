
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv = torch.nn.Conv2d(6, 12, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.maxpool(x1)
        v2 = self.conv(v1)
        v3 = v2 - 30
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 6, 32, 32)
