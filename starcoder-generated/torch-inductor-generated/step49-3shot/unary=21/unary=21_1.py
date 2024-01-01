
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 1.1 convolution + Tanh
        self.conv = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 16, 32, 32)
