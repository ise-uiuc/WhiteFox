
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_0 = torch.nn.Conv2d(in_channels=3, out_channels=7, kernel_size=1, bias=False)
        self.tanh_0 = torch.nn.Tanh()
    def forward(self, x0):
        x1 = self.conv_0(x0)
        x2 = self.tanh_0(x1)
        return x2
# Inputs to the model
x0 = torch.randn(1, 3, 10, 20)
