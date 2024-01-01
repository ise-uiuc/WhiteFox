
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout()

    def forward(self, x1):
        x2 = self.dropout(x1)
        x2 = torch.randn_like(x1)
        x3 = torch.rand_like(x1)
        x4 = x2 + x3
        x5 = torch.rand_like(x4)
        x6 = x4 + x5
        return x6
# Inputs to the model
x1 = torch.randn(10, 10)
