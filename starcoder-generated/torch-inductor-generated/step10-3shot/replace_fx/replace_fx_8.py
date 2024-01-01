
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        p1 = torch.nn.functional.dropout(x, p=0.5)
        v2 = torch.rand_like(x, dtype=torch.float)
        q1 = p1 + v2
        d1 = torch.nn.functional.dropout(p1, p=0.5)
        y = p1 + v2
        return q1 + d1
# Inputs to the model
x = torch.randn(3, 1, 10)
