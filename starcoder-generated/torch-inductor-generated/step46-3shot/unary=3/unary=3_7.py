
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(11, 10, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv2 = torch.nn.Conv2d(10, 25, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv3 = torch.nn.Conv2d(25, 10, (1, 1), stride=(1, 1), padding=(0, 0))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = v7 * 0.611098011577842
        v9 = v7 * 0.7969919788935442
        v10 = torch.erf(v9)
        v11 = v10 + 1
        v12 = v8 * v11
        v13 = self.conv3(v12)
        v17 = torch.conv1d()
        v18 = torch.conv2d(1, 1, (1, 1), stride=(1, 1), padding=(0, 0))
        return v13
# Inputs to the model
x1 = torch.randn(1, 11, 23, 23)
x2 = torch.randn(2, 4, 7, 7)
