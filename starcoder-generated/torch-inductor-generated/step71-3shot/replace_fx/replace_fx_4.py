
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.rand_like(x)
        t2 = F.dropout(x, p=0.5)
        t3 = torch.rand_like(t2)
        t4 = F.dropout(t1, p=0.5)
        t5 = torch.rand_like(t4)
        t6 = F.dropout(t5, p=0.5)
        return t2
# Inputs to the model
x = torch.randn(10, 2, 2)
