
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose3 = torch.nn.ConvTranspose2d(32, 32, 11, stride=11, padding=11)
    def forward(self, x1):
        v1 = self.conv_transpose3(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 32, 5, 5)
