
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 12, 1, stride=1, padding=1, groups=4)
    def forward(self, x):
        x = nn.functional.pad(x, (2, 2, 2, 2))
        x = F.relu(self.conv(x))
        return x
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
