
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose3d(128, 64, 1, padding=0, stride=4, output_padding=0, dilation=0, groups=1, bias=False)
        self.conv_transpose2 = torch.nn.ConvTranspose3d(64, 128, 3, padding=1, stride=2, output_padding=1, dilation=0, groups=1, bias=False)
        self.conv_transpose3 = torch.nn.ConvTranspose3d(128, 64, 1, padding=0, stride=2, output_padding=0, dilation=0, groups=1, bias=False)
        self.conv_transpose4 = torch.nn.ConvTranspose3d(64, 1, 5, padding=3, stride=1, output_padding=3, dilation=0, groups=1, bias=False)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_transpose2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv_transpose3(v4)
        v6 = torch.relu(v5)
        v7 = self.conv_transpose4(v6)
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 128, 16, 16, 16)
