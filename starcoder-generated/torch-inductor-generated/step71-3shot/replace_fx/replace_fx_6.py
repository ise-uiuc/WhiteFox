
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t0 = torch.sum(x, -1)
        t1 = torch.sum(t0, -1)
        t2 = torch.sum(t1, -1)
        t3 = torch.rand_like(t2)
        t4 = torch.sum(t2, 1)
        t5 = torch.sum(t4, -1)
        t6 = torch.sum(t3, -1)
        t7 = torch.sum(t0, 1)
        return t5
# Inputs to the model
x = torch.randn(1, 2, 3, 4)
