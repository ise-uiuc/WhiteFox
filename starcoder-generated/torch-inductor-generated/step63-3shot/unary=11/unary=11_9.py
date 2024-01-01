
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, 3, stride=1, padding=1)
        self.relu6 = torch.nn.ReLU6()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = self.relu6(v2)
        v4 = v3 / 6
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 2, 2)
