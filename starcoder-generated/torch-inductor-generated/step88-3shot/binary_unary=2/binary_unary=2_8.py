
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(3, 4, 3, stride=2, padding=1)
        self.c2 = torch.nn.Conv2d(3, 4, 3, stride=3, padding=2)
        self.c3 = torch.nn.Conv2d(3, 4, 2, stride=2, padding=2)
        self.c4 = torch.nn.Conv2d(3, 4, 1, stride=3, padding=3)
    def forward(self, x1):
        v1 = F.relu(self.c1(x1))
        v2 = F.relu(self.c2(x1))
        v3 = F.relu(self.c3(x1))
        v4 = F.relu(self.c4(x1))
        return v1 / v2 + v3 - v4
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
