
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1, dilation=2)
    def forward(self, x):
        v1 = self.conv(input)
        v2 = torch.nn.Tanh()(v1)
        return v2
# Inputs to the model
input = torch.randn(1, 3, 256, 256)
