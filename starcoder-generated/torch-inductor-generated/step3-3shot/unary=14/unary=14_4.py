
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1000)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 8, 1000, 64)
