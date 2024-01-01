
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(17, 3, 2, stride=1, padding=2, groups=2)
    def forward(self, x1, t1):
        v1 = self.conv(x1)
        if t1 == True:
            t1 = torch.mean(v1).reshape(v1.shape)
        v2 = v1 + t1
        return v2
# Inputs to the model
x1 = torch.randn(1, 17, 64, 64)
t1 = True
