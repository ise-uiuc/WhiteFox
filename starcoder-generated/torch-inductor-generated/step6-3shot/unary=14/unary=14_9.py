
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__():
        self.conv5_transpose = torch.nn.ConvTranspose2d(2, 2, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv5_transpose(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 128, 128)
