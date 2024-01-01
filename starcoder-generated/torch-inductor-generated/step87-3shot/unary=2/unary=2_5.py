
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose3d(3, 3, 2, stride=1, padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose3d(6, 2, 2, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 2, 4, 1)
