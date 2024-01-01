
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, x2, x3, x4):
        b = F.dropout(x)
        a = F.dropout(x)
        c = F.dropout(x)
        d = F.dropout(x)
        return d
# Inputs to the model
x = torch.randn(1)
x2 = torch.randn(1)
x3 = torch.randn(1)
x4 = torch.randn(1)
