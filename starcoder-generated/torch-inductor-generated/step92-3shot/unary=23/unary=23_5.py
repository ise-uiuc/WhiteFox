
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(7, 3, 3, stride=2, padding=1, groups=2)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(3, 3, 3, stride=1, padding=1, groups=2)
        self.identity = torch.nn.Identity()
    def forward(self, x):
        x1 = self.conv_transpose_1(x)
        x2 = self.conv_transpose_2(x1)
        return x2
# Inputs to the model
x1 = torch.randn(1, 7, 4, 4)
