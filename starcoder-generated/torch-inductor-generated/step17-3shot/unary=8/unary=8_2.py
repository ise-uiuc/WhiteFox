
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(10, 50, 5, groups=10)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(25, 64, 5, groups=5, stride=4)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(45, 83, 7, groups=25, stride=4)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = self.conv_transpose3(v2)
        v4 = v3 / 100
        return v4
# Inputs to the model
x1 = torch.randn(1, 10, 37, 30)
