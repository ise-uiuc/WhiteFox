
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(17, 17, 1, stride=1)
    def forward(self, x1, other=1, weight=1):
        if weight == 1:
            weight = torch.randn(self.conv.weight.shape)
        v1 = self.conv(x1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 17, 128, 128)
