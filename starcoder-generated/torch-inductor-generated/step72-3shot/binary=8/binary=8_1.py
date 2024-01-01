
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(29, 17)
        self.l2 = torch.nn.Linear(17, 19)
    def forward(self, x):
        v1 = self.l1(x)
        v2 = self.l2(x)
        return v1 * v2 + v1 + (1 + x).sum()
# Inputs to the model
x = torch.randn(2, 29)
