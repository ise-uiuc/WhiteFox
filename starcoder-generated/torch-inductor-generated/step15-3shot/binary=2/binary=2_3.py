
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.zero_weight = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 6.283185
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
