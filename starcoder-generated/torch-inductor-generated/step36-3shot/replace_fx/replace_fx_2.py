
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.5)
    def forward(self, x1):
        x2 = x1 ** (-3.2)
        x3 = self.dropout(x1) * x2
        x4 = torch.rand_like(x2)
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
