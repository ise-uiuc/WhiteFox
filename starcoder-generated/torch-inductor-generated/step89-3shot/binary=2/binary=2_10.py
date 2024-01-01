
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 5, stride=1, padding=2, dilation=4)
    def forward(self, x3):
        o0 = torch.t(x3)
        v1 = self.conv(o0)
        v2 = v1 - torch.tensor([[ 0.0747,  0.2693,  0.5305,  0.5493,  0.3789, -0.3719,  0.3202,  0.2067]])
        v3 = v2.type_as(o0).sum((0,1))
        v4 = v3.sigmoid()
        out = v4.view(1,4,1,1)
        return out
# Inputs to the model
x3 = torch.randn(4,4)
