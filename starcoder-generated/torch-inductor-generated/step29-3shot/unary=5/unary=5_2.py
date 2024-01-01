
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(3, 8, 3, stride=3, padding=1)
        self.maxpool = torch.nn.MaxPool2d(3,stride=2,padding=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(32, 8, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose2d(x1)
        v2 = self.maxpool(v1)
        v3 = self.conv_transpose(v2)
        v4 = v3 * 0.49999999999999994
        v5 = v3 * 0.5000000000000001
        v6 = torch.erf(v5)
        v7 = v6 + 1
        v8 = v4 * v7
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 30, 30)
