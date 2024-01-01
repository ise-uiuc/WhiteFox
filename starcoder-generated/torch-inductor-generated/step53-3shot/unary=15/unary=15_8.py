
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(30, 80, 15, stride=1, padding=7)
    def forward(self, x1):
        v1 = self.conv1(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 30, 1005, 535)
