
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t0 = torch.rand_like(x1)
        t1 = torch.rand_like(x1) * 1
        t3 = torch.nn.functional.dropout(t0, p=0.5, training=True)
        x2 = torch.rand_like(x1) * 0
        return x2 + t3
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
