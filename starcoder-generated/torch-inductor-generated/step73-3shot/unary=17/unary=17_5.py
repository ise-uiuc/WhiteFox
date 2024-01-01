
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 16, 1, stride=1)
        self.max = torch.nn.MaxPool2d(3, stride=2)
        self.conv1 = torch.nn.ConvTranspose2d(16, 8, 2, stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.max(v2)
        v4 = self.conv1(v3)
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(4, 3, 64, 64)
