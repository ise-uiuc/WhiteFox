
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 6, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(6, 8, 5, stride=1, padding=2)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 - v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 223, 223)
x2 = torch.randn(1, 2, 224, 224)
