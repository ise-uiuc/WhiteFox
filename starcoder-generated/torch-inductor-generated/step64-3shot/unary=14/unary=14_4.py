
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 1, 4, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        x2 = torch.add(v1, v1)
        return x2
# Inputs to the model
x1 = torch.randn(1, 1, 4, 4)
