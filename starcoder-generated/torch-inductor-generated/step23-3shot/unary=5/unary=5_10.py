
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 1, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.max_pool2d(x1)
        v2 = self.conv_transpose(v1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        return v7
# Inputs to the model
x1 = torch.randn(1, 4, 8, 8)
