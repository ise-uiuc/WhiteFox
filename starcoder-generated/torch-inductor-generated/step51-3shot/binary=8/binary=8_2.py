
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv8 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = nn.functional.relu(v1 + v2)
        v4 = self.conv3(x1).clamp(-50, 50)
        v5 = self.conv4(x2)
        v6 = v4.exp()
        v7 = v5.sum(axis=0)
        v8 = self.conv5(x1)
        v9 = self.conv6(x2)
        v10 = torch.nn.functional.elu(v8 + v9, alpha=20)
        v11 = torch.nn.functional.celu(v8 + v9, alpha=20)
        v12 = torch.nn.functional.hardshrink(v8 + v9, lambd=0.5)
        return torch.nn.functional.hardtanh(v8 + v9, minimum=0.2, maximum=0.8)
# Inputs to the model
n = 3
c = 3
h = 64
w = 64
x1 = torch.randn(n, c, h, w)
x2 = torch.randn(n, c, h, w)
