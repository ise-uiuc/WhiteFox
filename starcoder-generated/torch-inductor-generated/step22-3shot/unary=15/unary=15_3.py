
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 8, 5, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.maxpool2d(v1,2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 128, 128)
