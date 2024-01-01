
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(1, 1, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv_transpose_2(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 8, 8)
