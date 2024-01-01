
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(2, 2)
        self.l2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        x2 = torch.sigmoid(self.l1(x1.permute(0, 2, 1)))
        x3 = torch.sigmoid(self.l2(x2 * 7.0))
        x3 = x3.view([1, 2, 2])
        x3 = (x2 + x3) * 3.0
        return x3.permute(0, 2, 1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
