
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x):
        x = self.conv(x)
        x = x + 3
        x = torch.clamp_min(x, 0)
        x = torch.clamp_max(x, 6)
        x = torch.nn.functional.relu(x)
        x = x * 6
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
