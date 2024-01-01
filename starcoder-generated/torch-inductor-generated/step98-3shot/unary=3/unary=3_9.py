
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(34, 99, 8, stride=4, padding=2, output_padding=3)
        self.conv2 = torch.nn.Conv2d(99, 7, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = torch.sigmoid(x1)
        v2 = torch.conv1(v1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        v8 = self.conv2(v7)
        return v8
# Inputs to the model
x1 = torch.randn(6, 34, 71, 71)
