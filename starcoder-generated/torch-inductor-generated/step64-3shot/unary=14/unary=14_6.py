
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = v1.flatten(1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
