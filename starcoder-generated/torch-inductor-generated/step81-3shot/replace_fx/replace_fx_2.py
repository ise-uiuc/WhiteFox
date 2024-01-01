
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = torch.nn.functional.dropout(x1, p=0.4, training=False)
        t2 = torch.nn.functional.dropout(x1, p=0.5, training=True)
        t3 = torch.rand_like(t1)
        return t3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
