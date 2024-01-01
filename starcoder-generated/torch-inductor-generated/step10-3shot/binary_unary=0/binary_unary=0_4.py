
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        v4 = self.conv1(v3)
        v5 = v4 + x3
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
# Model end

# Model begins
class SSRU(nn.Module):
    def __init__(self, inp, oup, mid_features=None, has_mid_features=True, k_size=3, bn_relu=False):
        if not has_mid_features:
            mid_features = oup
        if mid_features is None:
            mid_features = max(int(oup/2), 8)

        super().__init__()

        self.conv = nn.Conv2d(int(inp), int(oup), int(k_size), 1, int((k_size - 1)/2), bias=True)

        self.bn = nn.BatchNorm2d(int(oup), eps=1e-03) if bn_relu else None
        self.sc = nn.Conv2d(int(inp), int(oup), 1, 1, bias=False)
        self.activ = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.softmax = nn.Softmax(dim=1)

        self.skip_proj = (int(inp) == int(oup))
        self.mid_proj = nn.Conv2d(int(inp), int(mid_features), 1, 1, 0, bias=False)
        self.g = nn.Conv2d(int(mid_features), int(mid_features), 1, 1, 0, bias=False)
        self.phi = nn.Conv2d(int(mid_features), int(oup), 1, 1, 0, bias=True)

        self.mp = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.sc(x)
        v1 = self.conv(x)
        v2 = self.softmax(-1) * v1
        v3 = v + v2

        if self.bn is not None:
            v3 = self.bn(v3)
        v4 = self.activ(v3)

        if not self.skip_proj:
            v1 = self.map_reduce(v1)

        v5 = self.g(v1)
        v6 = torch.sigmoid(self.softmax(-1) * self.phi(v5))
        v7 = v4 * v6
        v8 = v + v7

        return v8

    def map_reduce(self, fea):
        v = self.mp(fea)
        v1 = self.mid_proj(fea)
        v2 = self.softmax(-1)*v1
        v3 = v + v2
        return v3
# Inputs to the model
x = torch.randn(1, 224, 224, 3)
