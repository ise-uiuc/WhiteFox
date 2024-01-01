
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        with self.subgraph(device='cuda'):
            v1 = torch.mm(x1, x2)
            p = torch.addmm(x1[:, None], x2, v1)
        return v1 + p
# Inputs to the model
x1 = torch.randn(3, 3).cuda()
x2 = torch.randn(3, 3).cuda()
inp = torch.randn(3, 3).cuda()

