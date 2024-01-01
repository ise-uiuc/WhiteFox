
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=0)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv = torch.nn.Conv2d(5, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.softmax(v1)
        v3 = self.sigmoid(v1)
        v4 = v2 + v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
