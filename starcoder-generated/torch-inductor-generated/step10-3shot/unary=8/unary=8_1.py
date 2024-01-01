
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(24, 24, 3, stride=1, padding=0, bias=True, groups=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(24, 36, 1, stride=1, padding=0, bias=True, groups=1)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(36, 24, 3, stride=2, padding=0, bias=True, groups=1)
        self.conv_transpose4 = torch.nn.ConvTranspose2d(24, 3, 1, stride=2, padding=0, output_padding=1, bias=True, groups=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = self.conv_transpose3(v2)
        v4 = self.conv_transpose4(v3)
        v5 = v4 + 3
        return v5
# Inputs to the model
x1 = torch.randn(1, 24, 30, 30) 
