
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 12, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(12, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv2(self.conv1(x1))
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 32, 32)
