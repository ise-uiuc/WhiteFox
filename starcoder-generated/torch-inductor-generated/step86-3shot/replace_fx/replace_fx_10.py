
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        t1 = x2.squeeze_()
        t2 = t1 + 15
        t3 = t2 - 5
        x3 = F.dropout(t3, p=0.5)
        return x3
# Inputs to the model
x1 = torch.randn(1, 1, 1, 1)
