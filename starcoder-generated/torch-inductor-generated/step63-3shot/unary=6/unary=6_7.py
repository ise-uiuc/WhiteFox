
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 7, stride=1, padding=3)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1.mean()
        t3 = t1.view(128)
        t4 = torch.dot(t3, t2)
        t5 = t4 / 6.0
        return t5
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
