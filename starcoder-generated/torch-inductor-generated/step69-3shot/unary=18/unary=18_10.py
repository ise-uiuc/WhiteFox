
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        return torch.sigmoid(v1)
# Inputs to the model
x1 = torch.randn(1, 32, 128, 128)
