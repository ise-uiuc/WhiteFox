
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_9 = torch.nn.ConvTranspose2d(229, 9, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_9(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 229, 30, 10)
