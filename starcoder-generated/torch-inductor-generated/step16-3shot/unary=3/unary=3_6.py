
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 6, 5, stride=5, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 56, 56)
