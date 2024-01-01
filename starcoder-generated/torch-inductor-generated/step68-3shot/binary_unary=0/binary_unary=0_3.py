
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x):
        v0 = self.conv0(x)
        return v0
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
