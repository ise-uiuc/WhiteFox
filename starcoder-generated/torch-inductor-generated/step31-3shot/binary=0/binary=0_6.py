
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 2, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(2, 4, 1, stride=1, padding=1)
    def forward(self, x1, f1=1, f2=2):
        v1 = self.conv1(x1) + f1
        v2 = self.conv2(v1) + f2
        return v2
# Inputs to the model
x1 = torch.randn(3, 3, 64, 64)
