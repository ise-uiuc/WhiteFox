
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(16, 9, kernel_size=7, stride=1, padding=1, bias=None)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(9, 6, kernel_size=1, stride=1, padding=0)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(6, 2, kernel_size=1, stride=1, padding=0, bias=None)
        self.conv_transpose4 = torch.nn.ConvTranspose2d(2, 3, kernel_size=3, stride=1, padding=1, bias=None)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = self.conv_transpose3(v2)
        v4 = self.conv_transpose4(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 232, 321)
