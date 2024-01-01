
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 64, 1, stride=1, padding=0)
    def forward(self, x1):
        n1 = x1 - 3
        n2 = self.conv(n1)
        n3 = torch.add(n2, 3)
        n4 = torch.clamp_min(n3, 0)
        n5 = torch.clamp_max(n4, 6)
        n6 = n2 * n5
        n7 = n6 / 6
        n8 = x1 - 3
        n9 = self.conv(n8)
        n10 = torch.add(n9, 3)
        n11 = torch.clamp_min(n10, 0)
        n12 = torch.clamp_max(n11, 6)
        n13 = n9 * n12
        n14 = n13 / 6
        n15 = n7 + n14
        return n15
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
