
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(1, 5, 7, 1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(5, 16, 5, 1)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(16, 31, 20, 1)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_transpose_2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv_transpose_3(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
