
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv7 = torch.nn.Conv2d(2, 5, 7, stride=2, padding=3)
    def forward(self, x1):
        v1 = self.conv7(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 256, 256)
