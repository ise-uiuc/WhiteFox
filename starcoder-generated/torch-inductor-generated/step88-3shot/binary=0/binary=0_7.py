
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 9, 2, stride=2, padding=1)
    def forward(self, t1, other, **kwargs):
        t2 = torch.conv2d(i, torch.randn(3, 3, 4, 4), bias=None, stride=1, padding=1, dilation=2, groups=2) + other
# Inputs to the model
t1 = torch.randn(1, 8, 2, 2)
other = torch.randn(4)
kwarg1 = torch.randn(4)
