
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(11, 11, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 11, 540, 600)
