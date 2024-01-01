
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv_transpose_1_2 = torch.nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1)
        self.linear = torch.nn.Linear(3328, 37)
    def forward(self, x1):
        v1 = self.conv2(x1)
        v2 = self.conv_transpose_1_2(v1)
        v3 = torch.sigmoid(v2)
        v4 = v3 * v2
        v5 = v4.flatten(1)
        v6 = self.linear(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 16, 4, 4)
