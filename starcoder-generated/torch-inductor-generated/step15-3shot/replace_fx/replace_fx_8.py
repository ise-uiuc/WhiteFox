
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        p1 = torch.rand_like(x1)
        p2 = torch.rand_like(x1)
        p3 = torch.rand_like(x1)
        p4 = torch.rand_like(x1)
        p5 = torch.rand_like(x1)
        p6 = torch.rand_like(x1)
        p7 = p3 + p4
        p8 = torch.rand_like(x1)
        p9 = torch.rand_like(x1) + p7
        p10 = p5 + p6
        return p2 * p8 + p9 * p10
# Inputs to the model
