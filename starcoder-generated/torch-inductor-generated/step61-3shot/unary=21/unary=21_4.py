
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 3)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(x + v1)
        return v2
# Inputs to the model
x = torch.randn(1, 1, 224, 256)
