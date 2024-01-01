
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, 1, stride=1)
    def forward(self, x1):
        v1 = torch.tanh(self.conv1(x1))
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 256, 256)
