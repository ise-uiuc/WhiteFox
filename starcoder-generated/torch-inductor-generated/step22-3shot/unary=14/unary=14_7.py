
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(4, 4, 4, stride=4, padding=1)
    def forward(self, x2):
        v0 = x2.shape
        v1 = x2.reshape(736, 4)
        v2 = self.conv_transpose1(v1)
        v3 = torch.sigmoid(v2)
        v4 = v2 * v3
        v5 = v4.reshape(v0)
        return v5
# Inputs to the model
x2 = torch.randn(1, 1, 20, 20)
