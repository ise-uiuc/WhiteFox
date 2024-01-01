
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(21, 27, 2, stride=2)
        self.pool = torch.nn.MaxPool2d(2, stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.pool(v1)
        v3 = v1 * v2
        v4 = v3 / 7
        return v4
# Inputs to the model
x1 = torch.randn(1, 21, 44, 44)
