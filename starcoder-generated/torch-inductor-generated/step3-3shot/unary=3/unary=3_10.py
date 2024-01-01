
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 7, 1, stride=1, padding=2)
        self.conv2 = torch.nn.ConvTranspose2d(7, 4, 5, stride=2, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x2)
        v2 = self.conv2(v1)
        v3 = v2 * 0.7071067811865476
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
