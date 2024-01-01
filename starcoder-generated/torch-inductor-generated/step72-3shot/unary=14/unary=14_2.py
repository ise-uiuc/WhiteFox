
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, dilation=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = torch.tanh(v3)
        v5 = torch.abs(v1)
        v6 = torch.tanh(v5)
        v7 = self.conv_transpose2(v6)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        return v9
# Inputs to the model
x1 = torch.randn(1, 64, 284, 284)
