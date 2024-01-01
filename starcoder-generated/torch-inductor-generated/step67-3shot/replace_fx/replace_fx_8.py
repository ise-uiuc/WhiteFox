
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t0 = torch.pow(x, 3)
        dropout0 = F.dropout(t0, p=0.05)
        t1 = torch.pow(dropout0, 3)
        t2 = torch.rand_like(t0)
        t3 = torch.pow(t1, 3)
        dropout1 = F.dropout(t2, p=0.05)
        t4 = torch.pow(dropout1, 3)
        return t3
# Inputs to the model
x = torch.randn((10, 2, 2))
