

class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y1 = torch.randn(1)
        y2 = torch.randn(1)
        m = torch.nn.Dropout(0.3)
        y3 = m(y1)
        y4 = y3 * 2
        y5 = y2 + y4
        x1 = torch.nn.functional.dropout(x, p=0.1)
        z1 = x1 * 5
        z2 = z1 + y5
        return z2
# Inputs to the model
x1 = torch.randn(1, requires_grad=True)
