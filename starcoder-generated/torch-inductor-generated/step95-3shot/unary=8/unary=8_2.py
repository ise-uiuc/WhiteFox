
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 5, stride=2, bias=True, padding=2)
    def forward(self, x1):
        y1 = self.conv(x1)
        y2 = torch.floor(y1)
        return y2
# Inputs to the model
x1 = torch.randn(1, 1, 54, 94)
