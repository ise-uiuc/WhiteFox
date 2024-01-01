
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 2)
    def forward(self, x3):
        y = torch.relu(x3)
        y = y + 3
        return self.conv(y)
# Inputs to the model
x3 = torch.randn(1, 3, 3, 3)
