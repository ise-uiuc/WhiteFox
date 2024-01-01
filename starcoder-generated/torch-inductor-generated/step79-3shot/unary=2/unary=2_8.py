
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2 = torch.nn.ConvTranspose2d(6, 9, [0, 4], groups=3, bias=True)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(9, 99, 2, stride=2, padding=1, dilation=15, output_padding=2, bias=False)
    def forward(self, x1):
        v1 = self.conv_transpose2(x1)
        v2 = self.relu(v1)
        v3 = self.conv_transpose1(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 6, 30, 85)
