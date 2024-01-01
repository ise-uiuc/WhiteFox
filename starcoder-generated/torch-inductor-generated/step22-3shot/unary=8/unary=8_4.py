
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = torch.nn.MaxPool2d(2, stride=3)
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 4, 3, stride=4, padding=1, dilation=4)
    def forward(self, x1):
        v1 = self.max_pool(x1)
        v2 = self.conv_transpose(v1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=6)
        v6 = v2 * v5
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 6, 6)
