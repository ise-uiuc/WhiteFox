
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = torch.mm(x1, x2)
        t1 = torch.mm(x1, x2)
        t1 = torch.nn.functional.dropout(t1, p=0.1, training=False)
        t2 = torch.cat([t1, t1, t1, t1], 1)
        t2 = torch.cat([t2, t2, t2, t2], 1)
        t3 = torch.cat([t2, t2, t2, t2], 1)
        return torch.cat([t3, t3, t3, t3], 1)
# Inputs to the model
x1 = torch.randn(1, 4)
x2 = torch.randn(4, 4)
