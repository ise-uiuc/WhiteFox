
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, 3, stride=1, padding=0)
    def forward(self, x1, padding=0):
        v1 = self.conv(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 2, 4)
