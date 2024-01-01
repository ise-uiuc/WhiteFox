
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout()
    def forward(self, x):
        b = self.dropout(x)
        c = b.squeeze() * 0
        return self.dropout(c)
# Inputs to the model
x1 = torch.randn(100)
