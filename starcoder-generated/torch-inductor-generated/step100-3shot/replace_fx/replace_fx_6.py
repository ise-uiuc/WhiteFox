
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x1 = torch.rand_like(x)
        x2 = x * 2
        x3 = self.dropout(x1)
        return x1 + x2 + x3
# Inputs to the model
x = torch.randn(1, 2, 2)
