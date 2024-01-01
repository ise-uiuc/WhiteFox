
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(22, 3, 2, stride=2)
    def forward(self, x1):
        y1 = self.conv(x1)
        c1 = torch.tanh(y1)
        return c1
# Inputs to the model
x1 = torch.randn(1, 22, 169, 86)
