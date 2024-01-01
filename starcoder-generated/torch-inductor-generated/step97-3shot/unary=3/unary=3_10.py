
class ConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(63, 1, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        return v1
# Inputs to the model
x1 = torch.randn(16, 63, 19, 19)
