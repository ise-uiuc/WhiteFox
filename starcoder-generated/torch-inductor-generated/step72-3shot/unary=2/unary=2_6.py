
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(1, 4, 3, 1, 1, bias=True)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(4, 1, 3, 1, 1, bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(3, 1, 6, 6)
