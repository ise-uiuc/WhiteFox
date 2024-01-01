
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = x1 * x1
        x3 = x1 * x1
        t1 = torch.nn.functional.dropout(x2, p=0.5)
        x4 = t1 * x1
        x5 = torch.rand_like(x2)
        return x5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
