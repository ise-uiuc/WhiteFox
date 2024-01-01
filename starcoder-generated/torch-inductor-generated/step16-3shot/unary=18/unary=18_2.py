
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=48, kernel_size=3, stride=1, padding=1)
    def forward(self, x1, x2, x3):
        v1 = torch.sigmoid(self.conv1(x1))
        v2 = torch.softmax(v1, 1)
        v3 = v2 + 1.0
        v4 = torch.sigmoid(v2)
        m1 = torch.stack([v3, v4], 0)
        v5 = self.conv1(x2)
        v6 = torch.sigmoid(v5)
        v7 = torch.cat([m1, v6], 1)
        m2 = torch.sigmoid(v7)
        v8 = torch.cat([m2, x3], 1)
        return v8
# Inputs to the model
x1 = torch.randn(3, 3, 107, 107)
x2 = torch.randn(3, 3, 53, 53)
x3 = torch.randn(3, 3, 31, 31)
