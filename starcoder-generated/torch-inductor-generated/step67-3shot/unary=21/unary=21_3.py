
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_0 = torch.nn.Conv2d(in_channels=3, out_channels=7, kernel_size=3, stride=2, bias=False)
        self.tanh_0 = torch.nn.Tanh()
        self.conv_1 = torch.nn.Conv2d(in_channels=7, out_channels=12, kernel_size=3, stride=2, bias=False)
        self.tanh_1 = torch.nn.Tanh()
        self.conv_2 = torch.nn.Conv2d(in_channels=12, out_channels=2, kernel_size=3, stride=2, bias=False)
        self.tanh_2 = torch.nn.Tanh()
    def forward(self, x0):
        x1 = self.conv_0(x0)
        x2 = self.tanh_0(x1)
        x3 = self.conv_1(x2)
        x4 = self.tanh_1(x3)
        x5 = self.conv_2(x4)
        x6 = self.tanh_2(x5)
        return x6
# Inputs to the model
x0 = torch.randn(1, 3, 10, 20)
