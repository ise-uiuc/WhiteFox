
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.tensor([ 1.,  2.,
                                  3.,  4.])
        self.r0 = torch.tensor([ 5.,  6.])
    def forward(self, x1, x2):
        t0 = x1.permute(0, 2, 1)
        t1 = x2.permute(0, 2, 1)
        t2 = torch.bmm(t0, t1)
        t3 = self.t1.unsqueeze(2) * t2
        t4 = self.t1.unsqueeze(2) + t3
        t5 = t4 * t3
        t6 = torch.bmm(x1, t5)
        t7 = self.r0.unsqueeze(2) + x2
        return t7
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
