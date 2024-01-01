
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
    def forward(self, x1):
        a = self.linear(x1)
        b = x1 - 1
        c = F.dropout(a) * b
        d = a * b * F.dropout(b)
        return b + c * d
# Inputs to the model
x1 = torch.randn(1, 2, 4)
