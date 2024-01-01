
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 2, 3, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = - 7 * v1
        v3 = - 6 * v2
        v4 = - 5 * v3
        v5 = - 4 * v4
        v6 = - 3 * v5
        v7 = - 2 * v6
        v8 = - 1 * v7
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 5, 5)
