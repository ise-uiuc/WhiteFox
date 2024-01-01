
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(326, 117, 1, stride=1, padding=0, bias=False)
        self.relu = torch.nn.ReLU6()

    def forward(self, x4):
        v1 = self.conv_t(x4)
        v2 = v1 > 0
        v3 = v1 * -0.5315983825
        v4 = torch.where(v2, v1, v3)
        v5 = self.relu(v4)

        v6 = self.conv_t(v5)
        v7 = v6 > 0
        v8 = v6 * -0.5385402217
        v9 = torch.where(v7, v6, v8)

        v10 = self.relu(v9)
        return v10
# Inputs to the model
x4 = torch.randn(3, 326, 47, 22)
