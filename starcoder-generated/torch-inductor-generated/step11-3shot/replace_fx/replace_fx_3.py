
class m1(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
    def forward(self, x):
        a1 = torch.nn.functional.dropout(x, p=0)
        a2 = self.n
        a2 = torch.nn.functional.dropout(a2, p=0)
        return a1
# Inputs to the model
x1 = torch.randn(1)
m = m1(1)
