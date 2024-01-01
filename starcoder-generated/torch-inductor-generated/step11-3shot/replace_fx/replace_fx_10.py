
class A(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        a1 = torch.nn.functional.dropout(x, 0.2)
        a2 = torch.nn.functional.dropout(y, 0.2)
        return a1*a2
# Inputs to the model
x1 = torch.randn(1, 1, 1)
x2 = torch.randn(1, 1, 1)
