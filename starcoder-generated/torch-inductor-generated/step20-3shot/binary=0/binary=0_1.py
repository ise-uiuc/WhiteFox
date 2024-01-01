
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, 3, stride=1, padding=2)
    def forward(self, x1, zero=torch.zeros(1), one=torch.ones(1), two=[]):
        v1 = self.conv(x1)
        if len(two) == v1.shape[2]:
            v1 += one
        else:
            v1 += 0
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 56, 56)
