
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(33, 77, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(77, 97, 3, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(97, 15, 3, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(15, 9, 1, stride=1, padding=0)
        self.conv5 = torch.nn.ConvTranspose2d(9, 4, 3, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(1, 2, 1, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = torch.max(v3, dim=-1).values
        v5 = self.conv4(v4)
        v6 = torch.max(v5, dim=-1).values
        v7 = self.conv5(v6)
        v8 = v7 * 0.5
        v9 = v7 * 0.7071067811865476
        v10 = torch.erf(v9)
        v11 = v10 + 1
        v12 = v8 * v11
        v13 = self.conv6(v12)
        return v13
# Inputs to the model
x = torch.randn(8, 33, 27, 27)
