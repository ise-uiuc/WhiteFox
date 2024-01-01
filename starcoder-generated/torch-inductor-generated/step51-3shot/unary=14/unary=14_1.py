
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(34, 24, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 34, 25, 25)
