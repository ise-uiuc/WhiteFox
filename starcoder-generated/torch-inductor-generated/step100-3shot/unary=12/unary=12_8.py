
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1, stride=1, padding=0)
        self.softmax = torch.nn.Softmax()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.softmax(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
