
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 6, 3, stride=2, padding=1, output_padding=1, dilation=2)
    def forward(self, x):
        n1 = self.conv(x)
        n2 = torch.tanh(n1)
        return n2
# Inputs to the model
tensor = torch.randn(1, 3, 10, 10)
