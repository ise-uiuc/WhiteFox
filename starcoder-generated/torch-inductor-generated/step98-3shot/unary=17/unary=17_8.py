
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 48, 5, padding=1, groups=8, bias=False)
        self.conv1 = torch.nn.ConvTranspose2d(48, 3, 5, padding=1, groups=8, bias=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
