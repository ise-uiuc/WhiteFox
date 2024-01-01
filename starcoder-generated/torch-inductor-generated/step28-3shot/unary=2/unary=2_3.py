
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(1, 1, 1, 1, 1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(1, 1, 2, 2, (0, 1))
        self.conv_transpose3 = torch.nn.ConvTranspose2d(1, 1, 2, 2, (1, 0), 1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = self.conv_transpose3(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 1, 1)
