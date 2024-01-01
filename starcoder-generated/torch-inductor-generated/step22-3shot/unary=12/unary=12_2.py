
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv2 = torch.nn.Conv2d(1, 1, 2, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.sigmoid(v1)
        v3 = self.conv2(x1)
        v4 = v3 * v2
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2)
