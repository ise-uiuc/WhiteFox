
class Model(torch.nn.Module):
    def __init__(self, p1):
        super().__init__()
        self.p1 = p1
        self.dropout = torch.nn.Dropout(p1)
    def forward(self, x1):
        x2 = x1 ** -(self.p1 * 0.8)
        x3 = self.dropout(x2)
        x4 = torch.rand_like(x3)
        return (x4)
p1 = 1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
