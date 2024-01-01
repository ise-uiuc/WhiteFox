
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        a = torch.nn.functional.dropout(x1, p=0.1)
        b = F.dropout2d(x2, p=0.3, training=None)
        c = torch.nn.functional.dropout(b, p=0.5)
        d = torch.nn.functional.dropout(x3, p=0.6, training=False)
        return c + d
# Inputs to the model
x1, x2, x3 = torch.randn(3, 8, 8), torch.randn(3, 20, 20), torch.randn(3, 30, 30)
