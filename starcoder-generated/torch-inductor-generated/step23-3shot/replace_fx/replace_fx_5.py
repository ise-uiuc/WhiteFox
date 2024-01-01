
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(**{'p': 0.4})
    def forward(self, x):
        x1 = self.dropout(x)
        x2 = torch.rand_like(x1)
        return x2
# Inputs to the model
x = torch.randn(1, 2)
