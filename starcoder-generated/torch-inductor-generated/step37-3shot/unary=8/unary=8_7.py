
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convT = torch.nn.ConvTranspose2d(4, 4, 3, stride=1, dilation=1)
        self.relu6 = torch.nn.ReLU6()
    def forward(self, x1):
        v1 = self.convT(x1)
        v2 = v1 + 3
        v3 = (self.relu6)(v2)
        v4 = (self.relu6)(v3)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
