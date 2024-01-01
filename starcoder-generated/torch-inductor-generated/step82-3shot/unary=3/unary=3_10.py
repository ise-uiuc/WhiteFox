
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(2, 9, 1, stride=1, padding=0),torch.nn.Conv2d(9, 17, 2, stride=2, padding=0),torch.nn.Conv2d(17, 25, 3, stride=3, padding=1),torch.nn.Conv2d(25, 33, 4, stride=4, padding=2),torch.nn.Conv2d(33, 41, 5, stride=5, padding=3),torch.nn.Conv2d(41, 49, 6, stride=6, padding=4))
        self.conv2 = torch.nn.Conv2d(49, 2, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 33, 33)
