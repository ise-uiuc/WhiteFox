
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 8, 1, stride=1, padding=0, bias=False)
        self.conv2 = torch.nn.Conv2d(8, 1, 1, stride=1, padding=0, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = self.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
