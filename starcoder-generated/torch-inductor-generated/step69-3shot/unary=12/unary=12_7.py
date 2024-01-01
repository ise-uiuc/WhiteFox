
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 16, stride=4, dilation=4)
        self.pool = torch.nn.MaxPool2d(4, stride=4)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.pool(v1)
        v3 = self.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
