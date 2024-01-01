
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, 1, stride=1, padding=1, groups=2)
    def forward(self, x1, other1=False, other2=False):
        m1 = self.conv(x1)
        if other1 == False:
            other1 = m1
        elif other2 == False:
            other2 = m1
        m2 = m1 + other1
        m3 = m2 + other2
        return m3
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
