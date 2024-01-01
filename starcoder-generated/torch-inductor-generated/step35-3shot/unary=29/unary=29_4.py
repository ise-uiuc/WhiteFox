
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.padding = 1
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(3, 3, 3, stride=1, padding=self.padding)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(3, 3, 3, stride=1, padding=self.padding)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        self.padding += 1
        v2 = self.conv_transpose_2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 131, 131)
