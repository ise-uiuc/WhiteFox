
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(4,4), stride=2, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 1, kernel_size=(3,3), stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = torch.sigmoid(v2)
        v4 = torch.tanh(v3)
        v5 = torch.nn.functional.interpolate(v3, size=(112, 112), mode='bilinear')
        v6 = torch.nn.functional.interpolate(v4, size=(112, 112), mode='bilinear')
        v7 = v5 + v6
        v8 = self.conv2(v7)
        v9 = torch.sigmoid(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 112, 112)
