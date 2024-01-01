
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(32, 3, 1, stride=1, padding=1)
        self.conv1 = torch.nn.ConvTranspose2d(3, 32, 3, padding=1, stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 32, 32, 32)
