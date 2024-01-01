
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 3, padding=1, stride=2)
        self.conv1 = torch.nn.Conv2d(8, 16, 3, padding=1, stride=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1, stride=1)
        self.conv3 = torch.nn.ConvTranspose2d(32, 16, 3, padding=1, stride=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = torch.sigmoid(v7)
        return torch.flatten(v8, start_dim=1)
# Inputs to the model
x1 = torch.randn(32, 1, 32, 32)
