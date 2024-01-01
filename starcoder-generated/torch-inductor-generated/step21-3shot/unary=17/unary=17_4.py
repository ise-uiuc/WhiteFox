
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.ConvTranspose1d(1, 1, 3, stride=1, padding=0)
        self.conv1 = torch.nn.ConvTranspose2d(4, 64, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = torch.relu(v1)
        v3 = v2 + x1
        v4 = self.conv1(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 4, 100)
