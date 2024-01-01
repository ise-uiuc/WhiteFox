
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(4, 9, 3, stride=1, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(9, 9, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v11 = self.conv_transpose2(x1)
        v2 = torch.sigmoid(v1)
        v22 = torch.sigmoid(v11)
        v3 = v1 * v2
        v33 = v11 * v22
        return v3 * v33
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
