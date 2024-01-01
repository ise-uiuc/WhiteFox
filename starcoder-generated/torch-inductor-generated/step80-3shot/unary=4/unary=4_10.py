
class Model(torch.nn.Sequential):
    def __init__(self):
        super().__init__(
            torch.nn.Conv2d(3, 24, 3, stride=2, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(24, 48, 5, stride=3, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(48, 64, 1, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(64, 1)
        )
 
    def forward(self, x2):
        v1 = self.conv1
        v2 = v1(x1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        return v7
