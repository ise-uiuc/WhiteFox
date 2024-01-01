
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 9, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv3(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v3 = v3 * 10
        v4 = self.conv4(x1)
        v4 = v4 * 10
        return (v3, v4)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
