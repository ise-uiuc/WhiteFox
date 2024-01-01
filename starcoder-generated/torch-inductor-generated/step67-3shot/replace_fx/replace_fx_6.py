
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self, x):
        x = self.dropout(x)
        x = self.dropout(x)
        g = torch.nn.functional.gelu(x)
        e = g.exp()
        return e
# Inputs to the model
x = torch.randn(1)
