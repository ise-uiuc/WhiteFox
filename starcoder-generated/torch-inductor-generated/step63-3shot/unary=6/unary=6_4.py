
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=2, padding=0)
    def forward(self, x):
        y = self.conv1(x)
        return y
# Inputs to the model
x = torch.randn(1, 3, 500, 500)
