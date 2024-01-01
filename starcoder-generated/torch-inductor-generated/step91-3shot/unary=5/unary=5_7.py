
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(54, 34, 11, stride=2, padding=3)
        self.max_pool = torch.nn.MaxPool2d((7, 6), stride=2, padding=3)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.max_pool(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 54, 77, 77)
