
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 2, stride=2, padding=0)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 0.361456
        return v2
# Inputs to the model
x = torch.randn(1, 3, 255, 255)
