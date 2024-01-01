
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = torch.nn.functional.interpolate(t1, scale_factor=2)
        t3 = t2 - 127.0
        t4 = F.relu(t3)
        return t4
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
