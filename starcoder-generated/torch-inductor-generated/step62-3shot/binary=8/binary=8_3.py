
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 5, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 16, 5, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(3, 16, 5, stride=1, padding=2)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.conv3(x1)
        v4 = v1 + v2
        v5 = v3 - v2
        v6 = v3 * v2
        v7 = v3 / v2
        v8 = v1.exp()
        v9 = v4.exp()
        v10 = v5.exp().tanh()
        v11 = v6.exp().tanh()
        v12 = v7.exp().tanh()
        v13 = v8.ceil()
        v14 = v9.ceil()
        v15 = v13.floor()
        v16 = v14.floor()
        return v11

# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
