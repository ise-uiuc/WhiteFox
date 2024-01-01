
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.layer2 = torch.nn.Conv2d(1, 1, 3, stride=2, padding=0)
    def forward(self, x1):
        t1 = self.layer1(x1)
        t1_1 = t1 * 0.5
        t1_2 = t1 * 0.7071067811865475
        t1_3 = torch.erf(t1_2)
        t1_4 = t1_3 + 1
        t1_5 = t1_1 * t1_4
        t2 = self.layer2(t1_5)
        return t2
# Input to the model
x1 = torch.randn(1, 1, 526, 25)
