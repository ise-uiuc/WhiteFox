
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(8, 8, 3, stride=2, padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(8, 8, 1, stride=2, padding=0)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(8, 8, 5, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose2(x1)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose3(x1)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        v15 = v3 + v6 + v9
        return v15
# Inputs to the model
x1 = torch.randn(1, 8, 32, 32)
