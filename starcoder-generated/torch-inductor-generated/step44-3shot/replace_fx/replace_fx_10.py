
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = torch.rand_like(x1)
        t2 = t1 - t1 * x1
        t2 = F.dropout(x1, p=0.35, training=False)
        return t2
# Inputs to the model
x1 = torch.randn(1, 2, 3)
