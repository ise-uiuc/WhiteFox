
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3,  bias=False)
    def forward(self, x0):
        x1 = torch.tanh(self.conv_1(x0))
        return x1
# Inputs to the model
x0 = torch.randn(1, 4, 56, 56)
