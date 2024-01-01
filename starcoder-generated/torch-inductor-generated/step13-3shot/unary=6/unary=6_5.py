
class M1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=3)
    def forward(self, x1):
        e1 = self.conv1(x1)
        e2 = e1 + 3
        e3 = torch.clamp(e2, 0, 6)
        e4 = torch.cat([e1,e2,e3], 1)
        e5 = torch.sigmoid(e1 * e4 + 8)
        #e5 = e4
        e6 = e1 + e4
        e7 = e6 + e5 + 3
        e8 = e6 + e7
        return e5 + e8
# Inputs to the model
x1 = torch.zeros(2, 3, 64, 64)
