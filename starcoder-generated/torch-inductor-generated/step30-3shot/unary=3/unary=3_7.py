
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(32, 4, 7, stride=3, padding=3)
    def forward(self, x1):
        v1 = self.conv2(x1)
        v2 = v1 * 0.5
        return v2
# Inputs to the model
x1 = torch.randn(1, 32, 20, 20)
