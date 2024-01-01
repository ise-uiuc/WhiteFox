
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(3, 3, 2, stride=1, padding=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(3, 3, 2, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.relu6(v1)
        v3 = self.conv_transpose_2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 25, 25)
