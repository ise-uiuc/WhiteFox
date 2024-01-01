
class Model(torch.nn.Module):
    def __init__(self, conv_stride=4, conv_padding=4, fc_input_channels=4096):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=conv_stride, padding=conv_padding, groups=8)
        self.fc = torch.nn.Linear(fc_input_channels, 10)
    def forward(self, x1):
        v1 = self.conv(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
