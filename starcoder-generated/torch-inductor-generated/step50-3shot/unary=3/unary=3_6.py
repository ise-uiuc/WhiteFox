
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.ConvTranspose2d(3, 2, 1, stride=1, padding=0)
        self.conv3 = torch.nn.ConvTranspose2d(2, 2, 1, stride=1, padding=0)
        self.conv5 = torch.nn.ConvTranspose2d(2, 24, 2, stride=1, padding=0)
        self.conv6 = torch.nn.ConvTranspose2d(24, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv2(x1)
        v2 = self.conv3(v1)
        v3 = self.conv5(v2)
        v4 = v3 + 0.5
        v5 = v3 + 0.7071067811865476
        v6 = torch.erf(v5)
        v7 = v6 + 1
        v8 = v4 * v7
        v9 = self.conv6(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
