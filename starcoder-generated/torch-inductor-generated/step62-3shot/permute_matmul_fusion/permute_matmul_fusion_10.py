
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = x1.permute(0, 2, 1)
        t2 = x2.permute(0, 2, 1)
        vt = torch.dot(t1, torch.bmm(torch.bmm(t1, t2), t1))[0][0]
        return vt
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
