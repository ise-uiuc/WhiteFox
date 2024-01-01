
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(5, 25, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 5, 9, 9)
