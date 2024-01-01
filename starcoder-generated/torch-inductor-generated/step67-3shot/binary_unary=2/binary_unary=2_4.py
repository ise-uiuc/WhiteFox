
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 2, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 2, 2, stride=2, padding=1)
        self.deconv1 = torch.nn.ConvTranspose2d(2, 16, 2, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.deconv1(v2)
        v4 = v3 - 0.4
        v5 = F.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
