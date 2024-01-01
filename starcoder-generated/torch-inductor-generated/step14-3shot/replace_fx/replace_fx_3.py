
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout()
    def forward(self, x1):
        x2 = self.dropout(x1)
        x3 = torch.rand_like(x1)
        x4 = x3 * x2
        x5 = 1.0 / (x1 + x4)
        return x5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
