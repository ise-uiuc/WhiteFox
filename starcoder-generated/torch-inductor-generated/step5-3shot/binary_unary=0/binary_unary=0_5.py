
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 5, 3, stride=2, padding=1)
    def forward(self, x1, x2, x3):
        v1 = self.conv(x1)
        v2 = v1 + x2
        v3 = v2[:, :, :, :] + v2[:, :, :, :]
        v4 = torch.relu(v3)
        return v4[:, :, :, :]
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
x2 = torch.randn(2, 5, 19, 19)
x3 = torch.randn(2, 5, 1, 1)
