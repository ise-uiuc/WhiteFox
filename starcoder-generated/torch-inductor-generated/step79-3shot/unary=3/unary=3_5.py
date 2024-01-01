
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, stride=1, padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(3, 9, 2, stride=2, padding=0)
        self.conv3 = torch.nn.Linear(in_features=1, out_features=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = v7.reshape((1, 9))
        v9 = self.conv3(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 1, 121, 121)
