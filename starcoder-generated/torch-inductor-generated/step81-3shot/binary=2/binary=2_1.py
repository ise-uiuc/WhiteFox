
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=2, padding=2)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.sigmoid(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
