
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.deconv = torch.nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.deconv(v1)
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(2, 1, 4, 4)
