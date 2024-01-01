
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 3, padding=1, groups=3)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        y = self.relu(x)
        return y
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
