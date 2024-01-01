
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 7, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.nn.functional.interpolate(v1, size=[250, 250], mode='bilinear')
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 150, 150)
