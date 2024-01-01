
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.sigmoid(F.relu6(self.conv(x1)))
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
