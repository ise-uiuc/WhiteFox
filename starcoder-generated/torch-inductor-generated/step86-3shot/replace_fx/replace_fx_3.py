
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = x1 + 1
        t2 = torch.rand_like(t1)
        t3 = t1 * t2
        x = torch.nn.functional.dropout(t3)
        return  t1 + 2.0
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = 1
