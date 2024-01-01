
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2 = torch.nn.ConvTranspose2d(10, 9, 4, stride=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(9, 8, 4, stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose2(x1)
        v2 = self.conv_transpose(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 10, 16, 16)
