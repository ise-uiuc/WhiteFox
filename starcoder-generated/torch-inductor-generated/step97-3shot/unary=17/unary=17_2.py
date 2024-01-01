
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(1, 64, (8, 8), stride=1, padding=6, dilation=1)
        self.conv1 = torch.nn.ConvTranspose2d(64, 32, (8, 8), stride=1, padding=5, dilation=1)
        self.conv2 = torch.nn.ConvTranspose2d(64, 16, (8, 8), stride=1, padding=4, dilation=1)
        self.conv3 = torch.nn.ConvTranspose2d(64, 4, (8, 8), stride=1, padding=3, dilation=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
