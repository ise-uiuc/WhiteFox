
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(5120, 1600, 7, stride=1, padding=3)
        self.relu_1 = torch.nn.ReLU(inplace=False)
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(3300, 300, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_4(x1)
        v2 = self.relu_1(v1)
        v3 = self.conv_transpose_5(v2)
        return v3
# Inputs to the model
x1 = torch.randn(4, 5120, 9, 9)
