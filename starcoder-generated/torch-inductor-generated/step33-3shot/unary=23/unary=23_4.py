
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 2, kernel_size=15, stride=14, padding=5, dilation=4)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(2, 3, kernel_size=16, stride=37, padding=29, dilation=14)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv_transpose2(v2)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 62, 33)
