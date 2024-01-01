
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 6, 4, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(6, 1, 1, stride=1, padding=0)
        self.max_pool = torch.nn.MaxPool2d(1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = self.max_pool(v7)
        v9 = v8.view(-1, 12)
        return v9
# Inputs to the model
x1 = torch.randn(1, 2, 19, 47)
