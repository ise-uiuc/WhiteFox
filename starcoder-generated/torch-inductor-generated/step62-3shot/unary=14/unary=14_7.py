
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2 = torch.nn.ConvTranspose2d(216, 108, 3, stride=1, padding=1)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(108, 54, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose2(x1)
        v2 = self.conv_transpose3(v1)
        v3 = torch.sigmoid(v2)
        v4 = v2 * v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 216, 128, 128)
