
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 4, 3, stride=1, padding=1)
    def forward(self, x):
        y = self.conv1(x)
        return y
# Inputs to the model
x = torch.randn(1, 16, 59, 59)
