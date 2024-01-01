
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 4, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        l1 = 4 + v1
        lb1 = l1.clamp(min=0, max=6)
        l2 = 6 + lb1
        ll2 = l2.clamp(min=0, max=12)
        l3 = 8 + ll2
        ll3 = l3.clamp(min=0, max=16)
        l4 = 16 + ll3
        ll4 = torch.clamp_max(l4, 7)
        l5 = 16 + ll4
        ll5 = l5.clamp(min=0)
        l6 = 16 + ll5
        l7 = l6 / 8
        return l7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
