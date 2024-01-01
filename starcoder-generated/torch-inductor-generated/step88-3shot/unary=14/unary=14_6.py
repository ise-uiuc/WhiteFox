
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose3_2 = torch.nn.ConvTranspose3d(2, 1, 2, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose3_2(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(2, 2, 3, 4, 5)
