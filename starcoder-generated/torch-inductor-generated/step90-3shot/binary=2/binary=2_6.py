
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(48, 64, 1, stride=(1,1), padding=0, dilation=1, groups=1)
    def forward(self, x):
        t1 = self.conv1(x)
        t2 = t1 - 10
        return t2
# Inputs to the model
x = torch.randn(1, 48, 48, 48)
