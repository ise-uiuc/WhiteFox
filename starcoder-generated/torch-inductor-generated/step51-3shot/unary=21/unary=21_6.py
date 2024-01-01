
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1, stride=1, bias=False)
    def forward(self, x0):
        x1 = self.conv_1(x0)
        x2 = torch.tanh(x1)
        return x2
# Inputs to the model
x0 = torch.randn(1, 16, 64, 64)
