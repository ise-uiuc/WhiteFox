
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1, groups=32)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu6_(v1)
        v3 = torch.softmax(v2, dim=1)
        v4 = torch.sigmoid(v3)
        v5 = v4 + 1
        v6 = v5 - 1
        v7 = torch.flatten(v6, 1, 3)
        v8 = v6 - 1
        v9 = v8 / 1
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
