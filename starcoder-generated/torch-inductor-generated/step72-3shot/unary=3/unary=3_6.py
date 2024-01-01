
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 3, 1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = x1 * 0.5
        x3 = x1 * 0.7071067811865476
        x4 = F.relu(x3)
        x5 = x4 + 1
        x6 = x2 * x5
        y = self.conv2(x6)
        return y
# Inputs to the model
x1 = torch.randn(1, 3, 72, 72)
