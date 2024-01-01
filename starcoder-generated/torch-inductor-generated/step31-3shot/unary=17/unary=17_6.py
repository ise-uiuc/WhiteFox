
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(2, 1, 2)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(1, 1, 2)
        self.conv_2 = torch.nn.Conv2d(1, 1, 2)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.conv_transpose_1(v1)
        v3 = torch.relu(v2)
        v4 = self.conv_2(v3)
        return v4
# Inputs to the model
x1 = torch.randn(2, 2, 3, 3)
