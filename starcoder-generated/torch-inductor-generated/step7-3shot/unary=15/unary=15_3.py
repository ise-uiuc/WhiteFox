
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 256, 1, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(3, 1, 28, 28)
