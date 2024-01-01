
class ModelTanh3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.conv_2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2, dilation=1, bias=False)
        self.conv_3 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=2, padding=3, dilation=1, bias=False)
    def forward(self, x0):
        x1 = self.conv_1(x0)
        x1 = F.tanh(x1)
        x2 = self.conv_2(x0)
        x2 = F.tanh(x2)
        x3 = self.conv_3(x0)
        x3 = F.tanh(x3)
        return 
# Inputs to the model
x0 = torch.randn(1, 3, 28, 28)
