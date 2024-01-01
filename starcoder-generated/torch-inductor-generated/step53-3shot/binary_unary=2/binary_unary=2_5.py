
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv = torch.nn.Conv2d(16, 1, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.maxpool(x1)
        v2 = self.conv(v1)
        v3 = v2 - 128
        v4 = F.relu(v3)
        v5 = torch.squeeze(v4, -1)
        return v5
# Inputs to the model
x1 = torch.randn(1, 16, 76, 76)
