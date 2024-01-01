
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 4, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = nn.ReLU()(x)
        v2 = self.conv1(v1)
        v3 = nn.ReLU()(v2)
        v4 = self.conv2(v3+v3+v3+v3+v3+v3+v3+v3+v3+v3+v3+v3+v3+v3+v3+v3+v3+v3)
        v5 = nn.ReLU()(v4)
        vout = v4 + 1
        return vout
# Inputs to the model
x1 = torch.randn(16, 3, 224, 224)
