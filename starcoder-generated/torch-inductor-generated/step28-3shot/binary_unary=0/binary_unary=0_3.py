
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 7, stride=1, padding=3)
    def forward(self, x):
        v1 = self.conv(x)
        return v1
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
