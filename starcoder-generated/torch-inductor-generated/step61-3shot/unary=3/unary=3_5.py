
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(25, 2, 5, stride=5, padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(2, 5, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(5, 1, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        v8 = self.conv3(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 25, 68, 26)
