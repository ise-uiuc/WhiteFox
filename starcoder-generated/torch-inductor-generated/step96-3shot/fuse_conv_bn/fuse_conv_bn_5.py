
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Conv2d(3, 3, 1, bias=True)
        self.layer2 = torch.nn.Conv2d(3, 3, 3, bias=False)
    def forward(self, x3):
        y3 = self.layer1(x3) + self.layer2(y3)
# Inputs to the model
x3 = torch.randn(1, 3, 6, 6)
