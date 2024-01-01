
class ReLUAfterConcat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
    def forward(self, x):
        y = self.conv0(x)
        y = torch.cat((y, y), dim=1).view(32, -1).relu()
        y = self.conv1(x)
        y = torch.cat((y, y), dim=1).view(32, -1).relu()
        y = self.conv2(x)
        y = torch.cat((y, y), dim=1).view(32, -1).relu()
        return y.tanh()
# Inputs to the model
x = torch.randn(2, 1, 32, 32)
