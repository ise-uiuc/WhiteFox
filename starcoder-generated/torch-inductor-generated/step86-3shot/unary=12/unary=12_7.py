
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, (5, 5), stride=2, padding=2, bias=False)
    def forward(self, x1):
        v3 = self.conv(x1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
