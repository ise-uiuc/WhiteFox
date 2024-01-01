
class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
    def forward(self, x1, x2, x3, x4, x5, x6):
        t1 = torch.cat([x1, x2, x3, x4, x5, x6], dim = 1)
        t2 = t1[:, 0:18446744073709551615]
        t3 = t2[:, 0:size]
        t4 = torch.cat([t1, t3], dim = 1)
        return t4

# Input to the model
x1 = torch.randn(2, 1, 10, 2)
x2 = torch.randn(2, 2, 10, 2)
x3 = torch.randn(2, 4, 10, 2)
x4 = torch.randn(2, 8, 10, 2)
x5 = torch.randn(2, 16, 10, 2)
x6 = torch.randn(2, 98, 10, 2)

