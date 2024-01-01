
class ModelTan(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_0 = torch.nn.Conv2d(in_channels=7, out_channels=7, kernel_size=(3, 5), groups=7, bias=False)
        self.tanh_0 = torch.nn.Tanh()
    def forward(self, x0):
        x1 = self.conv_0(x0)
        x1 = self.tanh_0(x1)
        return x1
# Inputs to the model
x0 = torch.randn(1, 7, 112, 112)
