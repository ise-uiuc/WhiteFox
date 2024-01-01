
class Model_1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_depthwise = torch.nn.ConvTranspose2d(3, 16, 2, groups=3, padding=1, bias=False, stride=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 64, 3, padding=1, stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose_depthwise(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_transpose(v2)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
