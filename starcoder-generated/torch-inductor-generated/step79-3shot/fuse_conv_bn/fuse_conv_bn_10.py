
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Conv2d(1, 1, 1)
        self.l2 = torch.nn.Conv2d(1, 1, 1)
        torch.manual_seed(0)
        b1 = torch.zeros(1, 1, 1, 1).fill_(0.5)
    def forward(self, x1):
        y1 = self.l1(x1)
        y2 = self.l2(x1)
        y = 0.5 * ((y1 + y2) + b1 - 0.05)
        return y
# Inputs to the model
x1 = torch.randn(1, 1, 3, 3).fill_(0.5)
