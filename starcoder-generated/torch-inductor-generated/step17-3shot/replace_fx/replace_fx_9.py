
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        y1 = self.linear(x2)
        x3 = torch.rand_like(x1)
        y2 = self.linear(x3)
        return y1.sum() + y2.sum()
# Inputs to the model
x1 = torch.randn(10)
