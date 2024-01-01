
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 1, 7, stride=1, padding=0, dilation=1, groups=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = v2.unsqueeze(dim=1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 64, 112, 112)
