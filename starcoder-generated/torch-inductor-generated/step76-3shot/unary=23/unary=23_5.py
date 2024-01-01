
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose0 = torch.nn.ConvTranspose2d(4, 8, kernel_size=3, stride=2, padding=1)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose0(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv_transpose1(v2)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 4, 7, 7)
