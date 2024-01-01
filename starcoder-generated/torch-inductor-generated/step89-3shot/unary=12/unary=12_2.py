
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, bias=False)
        self.add = torch.add
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.add(v1, x1)
        v3 = self.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 14, 14)
