
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(12, 6, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = nn.Sigmoid()(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 12, 64, 64)
