
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(40, 10, 1)
        self.convt = torch.nn.ConvTranspose2d(10, 40, 1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = self.convt(v1)
        v4 = self.convt(v2)
        v5 = v3 + v4
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 320, 640)
