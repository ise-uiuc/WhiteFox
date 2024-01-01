
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = x1 * x1
        t1 = torch.nn.functional.dropout(x1, p=0.3)
        x2 = torch.rand_like(t1)
        x3 = torch.rand_like(t1)
        return x2 + x3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
