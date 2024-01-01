
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 66, 3, stride=1, dilation=6, padding=12)
    def forward(self, x1):
        v1 = self.conv(x1)
        return v1.mean(dim=(-1, -2), keepdim=True)
# Inputs to the model
x1 = torch.randn(2, 3, 112, 112)
