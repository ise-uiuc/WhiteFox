
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 6, 1, stride=1, padding=1)
    def forward(self, x1, conv2_weight=1):
        v1 = self.conv(x1)
        v2 = v1 + conv2_weight
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
