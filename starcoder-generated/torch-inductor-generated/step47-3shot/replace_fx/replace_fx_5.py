
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a = torch.nn.functional.dropout(x, p=0.5)
        b = torch.nn.functional.dropout(a, p=0.5)
        c = torch.nn.functional.dropout(b, p=0.5)
        d = torch.nn.functional.dropout(c, p=0.5)
        return d
# Inputs to the model
x1 = torch.randn(1, 2, 3)
