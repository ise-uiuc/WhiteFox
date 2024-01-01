
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t = x1.norm()
        t = torch.nn.functional.dropout(t)
        y1 = x1 * x2
        y2 = y1 / t
        return y2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1)
