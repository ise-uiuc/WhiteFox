
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 1)
        self.conv2 = torch.nn.ConvTranspose2d(32, 1, 3, padding=1, stride=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv1(v4)
        v6 = torch.relu(v5)
        return self.conv2(v6) + x1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
