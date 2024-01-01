
class Model(torch.nn.Module):
    def __init__(self, p1):
        super().__init__()
        self.p1 = p1
        self.dropout = torch.nn.Dropout(p1)
    def forward(self, x1):
        x2 = self.dropout(x1)
        x3 = torch.rand_like(x2)
        return x3
# Inputs to the model
x1 = torch.randn(3, 3, 3)
