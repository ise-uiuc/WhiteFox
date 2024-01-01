
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 48, 4, 26, groups=1)
        self.conv2 = torch.nn.ConvTranspose2d(64, 32, 48, 4, 26, groups=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1)
        v3 = self.conv2(v2)
        v4 = F.sigmoid(v3)
        v5 = v1 * v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 120, 120)
