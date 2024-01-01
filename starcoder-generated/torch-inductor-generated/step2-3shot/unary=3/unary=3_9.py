


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv_max_pooling = torch.nn.MaxPool2d(3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_max_pooling(v1)
        v3 = v2 * 0.5
        v4 = v2 * 0.5208333333333333
        v5 = v4 + 1
        v6 = v3 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
